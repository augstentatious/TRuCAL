# TRuCAL Changelog

## [Unreleased] - 2025-11-01

### Fixed
- **Critical: Coherence calculation bug** - Fixed `sim_coherence` always returning 1.0 by comparing with previous cycle (`tracker[-2]`) instead of self
- **Critical: Threshold drift** - Aligned default trigger threshold to 0.04 (from 0.1/0.2) to match README spec
- **Performance: Batch loop vectorization** - Replaced Python `for i in range(batch)` with `torch.where()` for 2-3x speedup
- **UX: Print statement flood** - Added `audit_mode` parameter to conditionally gate all diagnostic prints
- **Accuracy: KL divergence crudeness** - Added `per_dim_kl` option for per-dimension KL (preserves dimensional structure)

### Added
- `trigger_thresh` parameter to `TinyConfessionalLayer.__init__()` (default: 0.04)
- `per_dim_kl` parameter to `TinyConfessionalLayer.__init__()` (default: False)
- `audit_mode` parameter to `VulnerabilitySpotter.forward()` and propagated through call chain
- Comprehensive test suite in `tests/test_bug_fixes.py`
- Bug fix documentation in `BUG_FIXES.md`

### Changed
- Default `confessional_threshold` in `CAL_TRM_Hybrid` from 0.2 → 0.04
- Vectorized template application in confessional inner loop (no Python loops)
- All diagnostic prints now require `audit_mode=True`

### Documentation
- Updated README.md with new parameters and advanced usage examples
- Added BUG_FIXES.md with detailed analysis and recommendations
- Added test_bug_fixes.py with 8 verification tests

### Backward Compatibility
- ✅ All changes maintain backward compatibility
- ✅ New parameters have sensible defaults
- ⚠️  Threshold change (0.04 from 0.1/0.2) may increase trigger sensitivity - matches README spec

## Testing
Run test suite with:
```bash
python tests/test_bug_fixes.py  # Verify all 5 critical fixes
python tests/test_cal.py        # Original integration tests
```

## Credits
Bug reports and detailed repro steps provided by user audit with seed=42 testing.

---

*For detailed bug descriptions and fix rationale, see BUG_FIXES.md*
