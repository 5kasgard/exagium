/* Test: can the new mount API (fsopen/fsmount/move_mount) mount proc inside
 * an unprivileged user+pid namespace where mount(2) gets EPERM ("too revealing")? */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mount.h>
#include <sys/wait.h>
#include <sys/syscall.h>

static int fsopen_(const char *fs, unsigned flags) { return syscall(__NR_fsopen, fs, flags); }
static int fsconfig_(int fd, unsigned cmd, const char *key, const void *val, int aux) { return syscall(__NR_fsconfig, fd, cmd, key, val, aux); }
static int fsmount_(int fd, unsigned flags, unsigned attrs) { return syscall(__NR_fsmount, fd, flags, attrs); }
static int move_mount_(int from, const char *fpath, int to, const char *tpath, unsigned flags) { return syscall(__NR_move_mount, from, fpath, to, tpath, flags); }

int main(void) {
    /* we expect to run inside: unshare -Urmpf */
    int fsfd = fsopen_("proc", 0);
    if (fsfd < 0) { printf("fsopen failed: %s\n", strerror(errno)); return 1; }
    if (fsconfig_(fsfd, FSCONFIG_CMD_CREATE, NULL, NULL, 0) < 0) { printf("fsconfig create failed: %s\n", strerror(errno)); return 1; }
    int mfd = fsmount_(fsfd, 0, 0);
    if (mfd < 0) { printf("fsmount failed: %s\n", strerror(errno)); return 1; }
    if (mkdir("/tmp/mnt", 0755) && errno != EEXIST) { printf("mkdir: %s\n", strerror(errno)); return 1; }
    if (move_mount_(mfd, "", AT_FDCWD, "/tmp/mnt", MOVE_MOUNT_F_EMPTY_PATH) < 0) { printf("move_mount failed: %s\n", strerror(errno)); return 1; }
    printf("NEW-API-PROC-OK: /tmp/mnt is a fresh proc\n");
    /* classic mount(2) for comparison */
    mkdir("/tmp/mnt2", 0755);
    if (mount("proc", "/tmp/mnt2", "proc", 0, NULL) == 0) printf("old api also ok\n");
    else printf("old api: %s (expected)\n", strerror(errno));
    return 0;
}
