import Cocoa
import WebKit

/// Custom WKWebView subclass supporting native macOS Finder Drag & Drop
/// passing exact zero-copy filesystem paths (even for 1.5 TB sharded models!) to Rust.
class ForgeDropWebView: WKWebView {
    var onFilePathDropped: ((String) -> Void)?

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        return .copy
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let pb = sender.draggingPasteboard
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
           let first = urls.first {
            onFilePathDropped?(first.path)
            return true
        }
        return super.performDragOperation(sender)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKScriptMessageHandler, NSWindowDelegate {
    var window: NSWindow!
    var webView: ForgeDropWebView!
    var engineProcess: Process?
    let port: Int = 18795

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenu()
        startRustEngine()

        let frame = NSRect(x: 0, y: 0, width: 1440, height: 920)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "NALA-ModelForge-RS · Native Rust LLM Precision Studio"
        window.center()
        window.backgroundColor = NSColor(calibratedRed: 0.035, green: 0.043, blue: 0.059, alpha: 1.0)
        window.delegate = self

        let cfg = WKWebViewConfiguration()
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
        cfg.userContentController.add(self, name: "nalaOpenPanel")
        cfg.userContentController.add(self, name: "nalaOpenLoraPanel")

        webView = ForgeDropWebView(frame: frame, configuration: cfg)
        webView.uiDelegate = self
        webView.autoresizingMask = [.width, .height]
        webView.registerForDraggedTypes([.fileURL])
        webView.onFilePathDropped = { [weak self] path in
            self?.sendPathToJS(path)
        }

        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        waitForEngineAndLoad(attempts: 45)
    }

    func sendPathToJS(_ path: String) {
        if let data = try? JSONSerialization.data(withJSONObject: [path], options: []),
           let arrStr = String(data: data, encoding: .utf8) {
            let js = "if (window.onNativeDrop) { window.onNativeDrop((\(arrStr))[0]); }"
            DispatchQueue.main.async {
                self.webView.evaluateJavaScript(js, completionHandler: nil)
            }
        }
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        if message.name == "nalaOpenPanel" {
            let panel = NSOpenPanel()
            panel.title = "GGUF, Safetensors oder HuggingFace Modell-Ordner wählen"
            panel.allowsMultipleSelection = false
            panel.canChooseFiles = true
            panel.canChooseDirectories = true
            panel.beginSheetModal(for: window) { [weak self] resp in
                if resp == .OK, let url = panel.url {
                    self?.sendPathToJS(url.path)
                }
            }
        } else if message.name == "nalaOpenLoraPanel" {
            let panel = NSOpenPanel()
            panel.title = "LoRA Adapter Datei (.gguf / .safetensors) wählen"
            panel.allowsMultipleSelection = false
            panel.canChooseFiles = true
            panel.canChooseDirectories = true
            panel.beginSheetModal(for: window) { [weak self] resp in
                if resp == .OK, let url = panel.url {
                    if let data = try? JSONSerialization.data(withJSONObject: [url.path], options: []),
                       let arrStr = String(data: data, encoding: .utf8) {
                        let js = "if (window.onNativeLoraPick) { window.onNativeLoraPick((\(arrStr))[0]); }"
                        DispatchQueue.main.async {
                            self?.webView.evaluateJavaScript(js, completionHandler: nil)
                        }
                    }
                }
            }
        }
    }

    func setupMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Reload Studio", action: #selector(reloadPage), keyEquivalent: "r")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Quit NALA-ModelForge", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenuItem.submenu = appMenu

        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editMenuItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }

    @objc func reloadPage() {
        webView?.reload()
    }

    func startRustEngine() {
        guard let macOSDir = Bundle.main.executableURL?.deletingLastPathComponent() else { return }
        let engineURL = macOSDir.appendingPathComponent("nala-modelforge-engine")
        guard FileManager.default.isExecutableFile(atPath: engineURL.path) else { return }

        let proc = Process()
        proc.executableURL = engineURL
        proc.arguments = ["--headless", "--port", String(port)]
        var env = ProcessInfo.processInfo.environment
        env["NALA_MODELFORGE_PORT"] = String(port)
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:\(NSHomeDirectory())/bin"
        proc.environment = env
        do {
            try proc.run()
            self.engineProcess = proc
        } catch {
            NSLog("Failed to launch nala-modelforge-engine: \(error)")
        }
    }

    func waitForEngineAndLoad(attempts: Int) {
        let healthURL = URL(string: "http://127.0.0.1:\(port)/api/health")!
        let task = URLSession.shared.dataTask(with: healthURL) { _, response, _ in
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                DispatchQueue.main.async {
                    let appURL = URL(string: "http://127.0.0.1:\(self.port)/")!
                    self.webView.load(URLRequest(url: appURL))
                }
            } else if attempts > 0 {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.waitForEngineAndLoad(attempts: attempts - 1)
                }
            }
        }
        task.resume()
    }

    func applicationWillTerminate(_ notification: Notification) {
        engineProcess?.terminate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
