// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "S4TF_To_CoreML",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.6.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "S4TF_To_CoreML",
            dependencies: ["SwiftProtobuf"]),
        .testTarget(
            name: "S4TF_To_CoreMLTests",
            dependencies: ["S4TF_To_CoreML"]),
    ]
)
