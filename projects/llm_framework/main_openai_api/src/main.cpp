/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main()
{
    setenv("PYTHONPATH", "/opt/m5stack/lib/openai-api/site-packages", 1);

    if (access("/opt/m5stack/bin/ModuleLLM-OpenAI-Plugin/api_server.py", F_OK) == 0) {
        char *args[] = {(char *)"python3", (char *)"/opt/m5stack/bin/ModuleLLM-OpenAI-Plugin/api_server.py", NULL};
        if (execvp("python3", args) == -1) {
            perror("execvp");
            return 1;
        }
    }
    perror("_tokenizer.py miss");
    return 0;
}
