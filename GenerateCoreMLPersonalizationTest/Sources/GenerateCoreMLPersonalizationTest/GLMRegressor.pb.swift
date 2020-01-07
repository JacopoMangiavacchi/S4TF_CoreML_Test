// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: GLMRegressor.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// A generalized linear model regressor.
struct CoreML_Specification_GLMRegressor {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var weights: [CoreML_Specification_GLMRegressor.DoubleArray] = []

  var offset: [Double] = []

  var postEvaluationTransform: CoreML_Specification_GLMRegressor.PostEvaluationTransform = .noTransform

  var unknownFields = SwiftProtobuf.UnknownStorage()

  enum PostEvaluationTransform: SwiftProtobuf.Enum {
    typealias RawValue = Int
    case noTransform // = 0
    case logit // = 1
    case probit // = 2
    case UNRECOGNIZED(Int)

    init() {
      self = .noTransform
    }

    init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .noTransform
      case 1: self = .logit
      case 2: self = .probit
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    var rawValue: Int {
      switch self {
      case .noTransform: return 0
      case .logit: return 1
      case .probit: return 2
      case .UNRECOGNIZED(let i): return i
      }
    }

  }

  struct DoubleArray {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    var value: [Double] = []

    var unknownFields = SwiftProtobuf.UnknownStorage()

    init() {}
  }

  init() {}
}

#if swift(>=4.2)

extension CoreML_Specification_GLMRegressor.PostEvaluationTransform: CaseIterable {
  // The compiler won't synthesize support with the UNRECOGNIZED case.
  static var allCases: [CoreML_Specification_GLMRegressor.PostEvaluationTransform] = [
    .noTransform,
    .logit,
    .probit,
  ]
}

#endif  // swift(>=4.2)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_GLMRegressor: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".GLMRegressor"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "weights"),
    2: .same(proto: "offset"),
    3: .same(proto: "postEvaluationTransform"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.weights)
      case 2: try decoder.decodeRepeatedDoubleField(value: &self.offset)
      case 3: try decoder.decodeSingularEnumField(value: &self.postEvaluationTransform)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.weights.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.weights, fieldNumber: 1)
    }
    if !self.offset.isEmpty {
      try visitor.visitPackedDoubleField(value: self.offset, fieldNumber: 2)
    }
    if self.postEvaluationTransform != .noTransform {
      try visitor.visitSingularEnumField(value: self.postEvaluationTransform, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_GLMRegressor, rhs: CoreML_Specification_GLMRegressor) -> Bool {
    if lhs.weights != rhs.weights {return false}
    if lhs.offset != rhs.offset {return false}
    if lhs.postEvaluationTransform != rhs.postEvaluationTransform {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_GLMRegressor.PostEvaluationTransform: SwiftProtobuf._ProtoNameProviding {
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "NoTransform"),
    1: .same(proto: "Logit"),
    2: .same(proto: "Probit"),
  ]
}

extension CoreML_Specification_GLMRegressor.DoubleArray: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = CoreML_Specification_GLMRegressor.protoMessageName + ".DoubleArray"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "value"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedDoubleField(value: &self.value)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.value.isEmpty {
      try visitor.visitPackedDoubleField(value: self.value, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_GLMRegressor.DoubleArray, rhs: CoreML_Specification_GLMRegressor.DoubleArray) -> Bool {
    if lhs.value != rhs.value {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}