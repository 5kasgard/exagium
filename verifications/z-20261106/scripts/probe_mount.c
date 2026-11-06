#define _GNU_SOURCE
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>

static int fsopen_(const char *fs, unsigned flags) { return syscall(__NR_fsopen, fs, flags); }
static int fsconfig_(int fd, unsigned cmd, const char *key, const void *val, int aux) { return syscall(__NR_fsconfig, fd, cmd, key, val, aux); }
static int fsmount_(int fd, unsigned flags, unsigned attrs) { return syscall(__NR_fsmount, fd, flags, attrs); }
static int move_mount_(int from, const char *fpath, int to, const char *tpath, unsigned flags) { return syscall(__NR_move_mount, from, fpath, to, tpath, flags); }

int main(int argc, char **argv) {
    /* argv[1] = path to lazily detach first (or "none") */
    if (argc > 2 && strcmp(argv[1], "none")) {
        int rc = umount2(argv[1], MNT_DETACH);
        printf("umount2(%s, DETACH) = %d (%s)\n", argv[1], rc, rc ? strerror(errno) : "ok");
    }
    int fsfd = fsopen_("proc", 0);
    if (fsfd < 0) { printf("fsopen: %s\n", strerror(errno)); return 1; }
    if (fsconfig_(fsfd, FSCONFIG_CMD_CREATE, NULL, NULL, 0) < 0) { printf("fsconfig: %s\n", strerror(errno)); return 1; }
    int mfd = fsmount_(fsfd, 0, 0);
    if (mfd < 0) { printf("fsmount: %s  <- if EPERM, check dmesg for a fresh 'Mount too revealing'\n", strerror(errno)); return 1; }
    mkdir("/tmp/q", 0755);
    if (move_mount_(mfd, "", AT_FDCWD, "/tmp/q", MOVE_MOUNT_F_EMPTY_PATH) < 0) { printf("move_mount: %s\n", strerror(errno)); return 1; }
    printf("PROC-MOUNTED-OK via new api\n");
    return 0;
}
