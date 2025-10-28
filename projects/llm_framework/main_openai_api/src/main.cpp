/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    FILE *fp;
    char py_version[16]      = {0};
    char python_path[256]    = {0};
    const char *default_path = "/opt/m5stack/lib/openai-api/python3.11/site-packages";

    fp = popen("python3 -V 2>&1", "r");
    if (fp) {
        if (fgets(py_version, sizeof(py_version), fp)) {
            int major = 0, minor = 0;
            if (sscanf(py_version, "Python %d.%d", &major, &minor) == 2) {
                snprintf(python_path, sizeof(python_path), "/opt/m5stack/lib/openai-api/python%d.%d/site-packages",
                         major, minor);
            }
        }
        pclose(fp);
    }

    if (python_path[0] == '\0') strncpy(python_path, default_path, sizeof(python_path) - 1);

    setenv("PYTHONPATH", python_path, 1);

    const char *script_path = "/opt/m5stack/bin/ModuleLLM-OpenAI-Plugin/api_server.py";
    if (access(script_path, F_OK) == 0) {
        char *args[] = {(char *)"python3", (char *)script_path, NULL};
        if (execvp("python3", args) == -1) {
            perror("execvp");
            return 1;
        }
    }
    perror("_tokenizer.py miss");
    return 0;
}
