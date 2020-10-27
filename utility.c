#include <stdint.h>
#include <stdlib.h>

#if !defined(reallocarray)
	void *reallocarray(void *ptr, size_t nmemb, size_t size)
	{
		void* new_ptr = realloc(ptr, nmemb * size);

		return(new_ptr);
	}
#endif
