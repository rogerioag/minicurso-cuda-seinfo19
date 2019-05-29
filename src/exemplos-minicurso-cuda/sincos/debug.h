
#ifndef DEBUG_H
#define DEBUG_H

// #define DEBUG 1
// #define VERBOSE 1

#if defined(DEBUG) && DEBUG > 0
 #define TRACE(fmt, args...)	do{printf("[TRACE]: [%10s:%07d] in %s(): " fmt, \
    __FILE__, __LINE__, __func__, ##args); } while(0)

#else
 #define TRACE(fmt, args...) do{ } while (0)
#endif

#if defined(VERBOSE) && VERBOSE > 0
#define PRINT_FUNC_NAME printf("TRACE-FUNC-NAME: [%10s:%07d] is calling [%s()]\n",__FILE__, __LINE__, __FUNCTION__)
#else
#define PRINT_FUNC_NAME (void) 0
#endif

#endif /* DEBUG_H */
