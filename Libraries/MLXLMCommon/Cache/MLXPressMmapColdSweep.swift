// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

private typealias SafetensorsMmapAdviseLayerFn = @convention(c) (Int32, Int32) -> Int64
private typealias SafetensorsMmapAdviseRoutedFn = @convention(c) (Int32, Int32) -> Int64

/// Request-boundary controls for canonical mmap-backed routed weights.
///
/// `warmAll()` asks the kernel to read ahead every registered routed-expert
/// range. `cool(percent:)` returns the selected cold fraction to the file cache
/// policy. These calls operate on MLX's canonical tensor mappings through the
/// C ABI; they never allocate an overlay or change model values.
public enum MLXPressMmapRoutedResidency {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var resolved = false
    private nonisolated(unsafe) static var adviseRouted: SafetensorsMmapAdviseRoutedFn?

    @discardableResult
    public static func warmAll() -> Int64 {
        guard let advise = lookupAdviseRouted() else { return 0 }
        return advise(1, 0)
    }

    @discardableResult
    public static func cool(percent: Int) -> Int64 {
        guard let advise = lookupAdviseRouted() else { return 0 }
        return advise(0, Int32(max(0, min(100, percent))))
    }

    private static func lookupAdviseRouted() -> SafetensorsMmapAdviseRoutedFn? {
        lock.lock()
        defer { lock.unlock() }
        if resolved { return adviseRouted }
        resolved = true
        #if canImport(Darwin) || canImport(Glibc)
        guard let handle = dlopen(nil, RTLD_LAZY),
              let symbol = dlsym(handle, "mlx_safetensors_mmap_advise_routed")
        else {
            adviseRouted = nil
            return nil
        }
        adviseRouted = unsafeBitCast(symbol, to: SafetensorsMmapAdviseRoutedFn.self)
        return adviseRouted
        #else
        return nil
        #endif
    }
}

public enum MLXPressMmapColdSweep {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var resolved = false
    private nonisolated(unsafe) static var adviseLayer: SafetensorsMmapAdviseLayerFn?

    public static var isEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return isEnabledFlag(env["MLXPRESS_MMAP_LAYER_COLD_SWEEP"])
            || isEnabledFlag(env["JANGPRESS_MMAP_LAYER_COLD_SWEEP"])
    }

    public static func afterLayer(_ layer: Int, materialized output: MLXArray) {
        guard isEnabled else { return }
        MLX.eval(output)
        MLX.Memory.clearCache()
        _ = adviseLayerCold(layer)
    }

    @discardableResult
    public static func adviseLayerCold(_ layer: Int) -> Int64 {
        guard let advise = lookupAdviseLayer() else { return 0 }
        return advise(0, Int32(layer))
    }

    private static func lookupAdviseLayer() -> SafetensorsMmapAdviseLayerFn? {
        lock.lock()
        defer { lock.unlock() }
        if resolved {
            return adviseLayer
        }
        resolved = true
        #if canImport(Darwin) || canImport(Glibc)
        guard let handle = dlopen(nil, RTLD_LAZY),
              let symbol = dlsym(handle, "mlx_safetensors_mmap_advise_layer")
        else {
            adviseLayer = nil
            return nil
        }
        adviseLayer = unsafeBitCast(symbol, to: SafetensorsMmapAdviseLayerFn.self)
        return adviseLayer
        #else
        adviseLayer = nil
        return nil
        #endif
    }

    private static func isEnabledFlag(_ raw: String?) -> Bool {
        guard let raw = raw?.lowercased() else { return false }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }
}
