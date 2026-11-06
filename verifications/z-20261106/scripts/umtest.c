#define _GNU_SOURCE
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/mount.h>
#include <fcntl.h>
int main(int argc, char **argv) {
    int rc = umount2(argv[1], MNT_DETACH);
    printf("umount2(%s, MNT_DETACH) = %d errno=%d (%s)\n", argv[1], rc, errno, strerror(errno));
    if (argc > 2 && !strcmp(argv[2], "proc")) {
        rc = mount("proc", argv[3], "proc", 0, NULL);
        printf("mount proc = %d errno=%d (%s)\n", rc, errno, strerror(errno));
    }
    return 0;
}
