import Cocoa

let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "assets/icon_1024.png"
let size: CGFloat = 1024

let img = NSImage(size: NSSize(width: size, height: size))
img.lockFocus()

let rect = NSRect(x: 0, y: 0, width: size, height: size)
let bgPath = NSBezierPath(roundedRect: rect.insetBy(dx: 32, dy: 32), xRadius: 210, yRadius: 210)
let bgGrad = NSGradient(
    starting: NSColor(calibratedRed: 0.06, green: 0.08, blue: 0.11, alpha: 1.0),
    ending: NSColor(calibratedRed: 0.03, green: 0.04, blue: 0.06, alpha: 1.0)
)!
bgGrad.draw(in: bgPath, angle: -45)

NSColor(calibratedRed: 0.06, green: 0.72, blue: 0.50, alpha: 0.45).setStroke()
bgPath.lineWidth = 10
bgPath.stroke()

// Precision Tensor Prism Bars (representing 8-bit FAT coding core + 1-bit compressed layers)
let bars: [(CGFloat, CGFloat, CGFloat, NSColor)] = [
    (210, 620, 110, NSColor(calibratedRed: 0.06, green: 0.73, blue: 0.51, alpha: 1.0)),
    (360, 520, 110, NSColor(calibratedRed: 0.20, green: 0.83, blue: 0.60, alpha: 1.0)),
    (510, 340, 110, NSColor(calibratedRed: 0.96, green: 0.62, blue: 0.04, alpha: 0.95)),
    (660, 440, 110, NSColor(calibratedRed: 0.22, green: 0.74, blue: 0.97, alpha: 0.95))
]
for (x, h, w, col) in bars {
    let r = NSRect(x: x, y: 210, width: w, height: h)
    let p = NSBezierPath(roundedRect: r, xRadius: 26, yRadius: 26)
    col.setFill()
    p.fill()
}

let attrs: [NSAttributedString.Key: Any] = [
    .font: NSFont.monospacedSystemFont(ofSize: 96, weight: .heavy),
    .foregroundColor: NSColor(calibratedWhite: 0.94, alpha: 1.0)
]
let title = NSString(string: "FORGE·RS")
let ts = title.size(withAttributes: attrs)
title.draw(at: NSPoint(x: (size - ts.width) / 2.0, y: 80), withAttributes: attrs)

img.unlockFocus()

if let tiff = img.tiffRepresentation,
   let rep = NSBitmapImageRep(data: tiff),
   let png = rep.representation(using: .png, properties: [:]) {
    try png.write(to: URL(fileURLWithPath: outPath))
}
