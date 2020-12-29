#ifdef __INTELLISENSE__
/**
 * A work around <<< >>> for cuda kernels in CPP intellisense (hides some intellisense errors)
 */
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif  // __INTELLISENSE__