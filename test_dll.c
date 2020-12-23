#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void __declspec ( dllimport ) xray_image();

#ifdef __cplusplus
}
#endif

int main(void) {
    xray_image();
    return 0;
}