#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH

echo "Measuring LLVM-IR size ..."

rm -rf target
cargo rustc -- --emit=llvm-ir > /dev/null 2>&1
DEBUG_LLVM_IR_BYTES=$(printf "%'d" $(stat --printf="%s" target/debug/deps/*.ll))
echo "Debug LLVM-IR bytes:   ${DEBUG_LLVM_IR_BYTES}"

rm -rf target
cargo rustc --release -- --emit=llvm-ir > /dev/null 2>&1
RELEASE_LLVM_IR_BYTES=$(printf "%'d" $(stat --printf="%s" target/release/deps/*.ll))
echo "Release LLVM-IR bytes: ${RELEASE_LLVM_IR_BYTES}"

# Spacer
echo ""

hyperfine --min-runs 3 --prepare 'cargo clean' 'cargo build'

hyperfine --min-runs 3 --prepare 'cargo clean' 'cargo build --release'

# Does not show a large difference. Probably not representative of real world lto impact.
# CARGO_PROFILE_release_LTO=thin hyperfine --prepare 'cargo clean' 'cargo build --release'
