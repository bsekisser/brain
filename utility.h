#define T_LOG(_f, args...) \
	do { \
			printf("[%s:%04u] " _f, __FUNCTION__, __LINE__, ## args); \
	}while(0);

#define T(_f, args...) \
	do { \
		if(trace) \
			T_LOG(_f, ## args); \
	}while(0);

#define T_ERRNO_MSG() T_LOG("errno=%u, %m\n", errno)
#define T_ERR_ERRNO_MSG() T_LOG("err=%02i, errno=%u, %m\n", err, errno)

#define _UNUSED(_arg_) (void)(_arg_)

#if !defined(reallocarray)
	void *reallocarray(void *ptr, size_t nmemb, size_t size);
#endif

static inline float float_rand( float min, float max )
{
	float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
	return min + scale * ( max - min );      /* [min, max] */
}
