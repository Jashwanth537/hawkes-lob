// Binance WebSocket LOB event collector
// Streams: @bookTicker (OFI events) + @aggTrade (market-order events)
// Symbols: BTCUSDT, ETHUSDT
// Output:  <symbol>_events.csv  (appended, not overwritten on reconnect)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono;

// ── global stop flag ──────────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};

static void on_signal(int) { g_running = false; }

// ── helpers ───────────────────────────────────────────────────────────────────
static std::string now_str()
{
    auto tp = system_clock::now();
    auto t  = system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << '[' << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ']';
    return ss.str();
}

static int64_t now_us()
{
    return duration_cast<microseconds>(
               system_clock::now().time_since_epoch())
        .count();
}

// Sleep in 100 ms chunks so SIGINT wakes us up promptly.
static void interruptible_sleep(int seconds)
{
    for (int i = 0; i < seconds * 10 && g_running; ++i)
        std::this_thread::sleep_for(milliseconds(100));
}

// ── per-symbol state ──────────────────────────────────────────────────────────
struct BookState {
    double bid_px  = 0.0;
    double bid_qty = 0.0;
    double ask_px  = 0.0;
    double ask_qty = 0.0;
    bool   valid   = false;
};

struct Stats {
    std::atomic<int64_t> buy_mo{0};
    std::atomic<int64_t> sell_mo{0};
    std::atomic<int64_t> ofi{0};
    int64_t total() const { return buy_mo + sell_mo + ofi; }
};

// ── CSV writer ────────────────────────────────────────────────────────────────
// Thread-safe; flushes every 1000 events or every 5 seconds.
class CsvWriter {
public:
    explicit CsvWriter(const fs::path& path)
    {
        bool existed = fs::exists(path);
        file_.open(path, std::ios::app | std::ios::binary);
        if (!file_)
            throw std::runtime_error("Cannot open " + path.string());
        if (!existed)
            file_ << "timestamp_us,event_type,price,quantity,symbol\n";
        last_flush_ = steady_clock::now();
    }

    ~CsvWriter() { flush(); }

    void write(int64_t ts, int type, double price, double qty,
               const std::string& sym)
    {
        std::lock_guard<std::mutex> lk(mu_);
        // Enforce monotone timestamps: if two events (from bookTicker and
        // aggTrade sub-streams) have timestamps that differ by only a few µs
        // due to server/local clock offset, bump rather than distort ordering.
        if (ts <= last_ts_) ts = last_ts_ + 1;
        last_ts_ = ts;
        buf_ << ts << ',' << type << ','
             << std::fixed << std::setprecision(8)
             << price << ',' << qty << ',' << sym << '\n';
        ++pending_;

        auto now = steady_clock::now();
        bool age_exceeded =
            duration_cast<seconds>(now - last_flush_).count() >= 5;
        if (pending_ >= 1000 || age_exceeded)
            flush_locked();
    }

    void flush()
    {
        std::lock_guard<std::mutex> lk(mu_);
        flush_locked();
    }

private:
    void flush_locked()
    {
        auto s = buf_.str();
        if (!s.empty()) {
            file_ << s;
            file_.flush();
            buf_.str("");
            buf_.clear();
        }
        pending_    = 0;
        last_flush_ = steady_clock::now();
    }

    std::ofstream              file_;
    std::ostringstream         buf_;
    std::mutex                 mu_;
    int                        pending_    = 0;
    int64_t                    last_ts_    = 0;
    steady_clock::time_point   last_flush_;
};

// ── per-symbol collector ──────────────────────────────────────────────────────
class SymbolCollector {
public:
    SymbolCollector(std::string sym, const fs::path& outdir)
        : sym_(std::move(sym))
    {
        std::string low = sym_;
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);

        fname_ = low + "_events.csv";

        // Combined stream: bookTicker + aggTrade with microsecond event times.
        url_ = "wss://data-stream.binance.vision/stream?streams="
               + low + "@bookTicker/" + low
               + "@aggTrade&timeUnit=MICROSECOND";

        csv_ = std::make_unique<CsvWriter>(outdir / fname_);
    }

    void start() { thr_ = std::thread(&SymbolCollector::run, this); }
    void join()  { if (thr_.joinable()) thr_.join(); }
    void flush() { csv_->flush(); }

    const std::string& symbol()   const { return sym_;   }
    const std::string& filename() const { return fname_; }
    const Stats&       stats()    const { return stats_; }

private:
    // Opens one WebSocket session.
    // Returns true  → we connected and later disconnected normally.
    // Returns false → we never reached Open state.
    bool try_connect()
    {
        std::atomic<bool> opened{false};
        std::atomic<bool> closed{false};

        ix::WebSocket ws;
        ws.setUrl(url_);
        ws.disableAutomaticReconnection();

        ws.setOnMessageCallback(
            [this, &opened, &closed](const ix::WebSocketMessagePtr& m) {
                switch (m->type) {

                case ix::WebSocketMessageType::Open:
                    opened = true;
                    std::cout << now_str() << ' ' << sym_
                              << " connected. Writing to " << fname_ << '\n'
                              << std::flush;
                    break;

                case ix::WebSocketMessageType::Close:
                    closed = true;
                    break;

                case ix::WebSocketMessageType::Error:
                    closed = true;
                    std::cerr << now_str() << ' ' << sym_
                              << " WS error: " << m->errorInfo.reason << '\n';
                    break;

                case ix::WebSocketMessageType::Message:
                    on_message(m->str);
                    break;

                // Ping → IXWebSocket replies with pong automatically.
                default:
                    break;
                }
            });

        ws.start();

        // Wait up to 10 s for the connection to open.
        for (int i = 0; i < 100 && !opened && !closed && g_running; ++i)
            std::this_thread::sleep_for(milliseconds(100));

        if (!opened) {
            ws.stop();
            return false;
        }

        // Stay here until the server closes, we error out, or SIGINT.
        while (!closed && g_running)
            std::this_thread::sleep_for(milliseconds(100));

        ws.stop(); // synchronous: waits for the callback thread to finish
        return true;
    }

    void run()
    {
        std::cout << now_str() << " Connecting " << sym_ << " stream...\n"
                  << std::flush;

        int wait_sec   = 1;
        bool first_try = true;

        while (g_running) {
            if (!first_try) {
                std::cerr << now_str() << ' ' << sym_
                          << " reconnecting in " << wait_sec << " s...\n";
                interruptible_sleep(wait_sec);
                if (!g_running) break;
            }
            first_try = false;

            bool ok = try_connect();

            if (!g_running) break;

            if (ok) {
                std::cerr << now_str() << ' ' << sym_ << " disconnected.\n";
                wait_sec = 1; // reset backoff after a successful session
            } else {
                std::cerr << now_str() << ' ' << sym_
                          << " connection failed.\n";
                wait_sec = std::min(wait_sec * 2, 60);
            }
        }

        csv_->flush();
        std::cerr << now_str() << ' ' << sym_ << " collector stopped.\n";
    }

    // ── message dispatch ──────────────────────────────────────────────────────
    void on_message(const std::string& raw)
    {
        try {
            auto j = json::parse(raw);

            // Combined-stream envelope: {"stream":"...","data":{...}}
            if (!j.contains("stream") || !j.contains("data")) return;

            const auto& stream = j["stream"].get_ref<const std::string&>();
            const json& data   = j["data"];

            if (stream.find("@aggTrade") != std::string::npos)
                handle_agg_trade(data);
            else if (stream.find("@bookTicker") != std::string::npos)
                handle_book_ticker(data);

        } catch (const std::exception& e) {
            std::cerr << now_str() << ' ' << sym_
                      << " parse error: " << e.what() << '\n';
        }
    }

    // ── aggTrade → market order event ─────────────────────────────────────────
    void handle_agg_trade(const json& d)
    {
        // m=true  → buyer is the passive maker → aggressive seller hit the book
        //         → SELL market order  (event type 1)
        // m=false → buyer is aggressive → BUY market order (event type 0)
        bool buyer_maker = d.value("m", false);

        int64_t ts = (d.contains("E") && d["E"].is_number())
                         ? d["E"].get<int64_t>()
                         : now_us();

        double price = std::stod(d["p"].get<std::string>());
        double qty   = std::stod(d["q"].get<std::string>());
        int    type  = buyer_maker ? 1 : 0;

        csv_->write(ts, type, price, qty, sym_);

        if (buyer_maker) ++stats_.sell_mo;
        else             ++stats_.buy_mo;
    }

    // ── bookTicker → OFI event ────────────────────────────────────────────────
    void handle_book_ticker(const json& d)
    {
        double bid_px  = std::stod(d["b"].get<std::string>());
        double bid_qty = std::stod(d["B"].get<std::string>());
        double ask_px  = std::stod(d["a"].get<std::string>());
        double ask_qty = std::stod(d["A"].get<std::string>());

        // Seed state without emitting on the very first tick.
        if (!book_.valid) {
            book_ = {bid_px, bid_qty, ask_px, ask_qty, true};
            return;
        }

        // Price-level change rule:
        //   If the best-bid price changed, the old resting quantity is fully
        //   cancelled (treat prev_bid_qty as 0) and the new quantity is a
        //   fresh arrival. Same for the ask side.
        double d_bid = (bid_px != book_.bid_px) ? bid_qty
                                                : bid_qty - book_.bid_qty;
        double d_ask = (ask_px != book_.ask_px) ? ask_qty
                                                : ask_qty - book_.ask_qty;

        book_ = {bid_px, bid_qty, ask_px, ask_qty, true};

        double ofi = d_bid - d_ask;
        if (ofi == 0.0) return;

        int64_t ts = (d.contains("E") && d["E"].is_number())
                         ? d["E"].get<int64_t>()
                         : now_us();

        // OFI_POS (2): net pressure toward bid (buying interest)
        // OFI_NEG (3): net pressure toward ask (selling interest)
        int type = (ofi > 0.0) ? 2 : 3;
        csv_->write(ts, type, bid_px, std::abs(ofi), sym_);
        ++stats_.ofi;
    }

    std::string                sym_;
    std::string                fname_;
    std::string                url_;
    std::unique_ptr<CsvWriter> csv_;
    BookState                  book_; // only touched by WS callback thread
    Stats                      stats_;
    std::thread                thr_;
};

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    fs::path outdir = ".";
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--outdir" && i + 1 < argc)
            outdir = argv[++i];
    }

    if (!fs::exists(outdir) || !fs::is_directory(outdir)) {
        std::cerr << "Error: output directory does not exist: "
                  << outdir << '\n';
        return 1;
    }

    ix::initNetSystem();

    std::vector<std::unique_ptr<SymbolCollector>> collectors;
    collectors.push_back(std::make_unique<SymbolCollector>("BTCUSDT", outdir));
    collectors.push_back(std::make_unique<SymbolCollector>("ETHUSDT", outdir));

    for (auto& c : collectors) c->start();

    auto t_stats = steady_clock::now();
    auto t_flush = steady_clock::now();

    while (g_running) {
        std::this_thread::sleep_for(seconds(1));
        auto now = steady_clock::now();

        // Periodic flush so the 5-second guarantee holds even during quiet markets.
        if (duration_cast<seconds>(now - t_flush).count() >= 5) {
            for (auto& c : collectors) c->flush();
            t_flush = now;
        }

        if (duration_cast<seconds>(now - t_stats).count() >= 60) {
            t_stats = now;
            for (auto& c : collectors) {
                const auto& s = c->stats();
                std::cout << now_str() << ' ' << c->symbol() << ": "
                          << s.total()       << " events ("
                          << s.buy_mo.load() << " BUY_MO, "
                          << s.sell_mo.load()<< " SELL_MO, "
                          << s.ofi.load()    << " OFI)\n"
                          << std::flush;
            }
        }
    }

    std::cout << now_str() << " Shutting down...\n" << std::flush;
    for (auto& c : collectors) c->join();

    ix::uninitNetSystem();
    std::cout << now_str() << " Done.\n";
    return 0;
}
