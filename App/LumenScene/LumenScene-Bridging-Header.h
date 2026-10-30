// Bridging header: exposes the Cascade façade (C ABI over the
// Open CASCADE libraries) to Swift. Used only by StepImport.swift
// for STEP file import; everything else in the app is pure Swift.
#import "occt_facade.h"
