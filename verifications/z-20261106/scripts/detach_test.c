#define _GNU_SOURCE
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/stat.h>
/* detach listed paths (lazy), then try mounting proc at the last arg */
int main(int argc, char **argv) {
    for (int i = 1; i < argc - 1; i++) {
        int rc = umount2(argv[i], MNT_DETACH);
        printf("umount2(%s, DETACH) = %d errno=%d (%s)\n", argv[i], rc, errno, rc ? strerror(errno) : "ok");
    }
    mkdir(argc > 1 ? argv[argc-1] : "/tmp/q", 0755);
    int rc = mount("proc", argv[argc-1], "proc", 0, NULL);
    printf("mount proc at %s = %d errno=%d (%s)\n", argv[argc-1], rc, errno, rc ? strerror(errno) : "ok");
    return rc ? 1 : 0;
}
