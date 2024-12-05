from prompt_chain import multi_level_cot
import wget
def savePromptToFile(prompt):
    summary = multi_level_cot(prompt)
    with open("prompt.txt", "w") as f:
        f.write(summary)
    return summary

def readPromptFromFile(filepath):
    with open(filepath, "r") as f:
        return f.read()

def saveImageFromUrl(filename):
    wget.download(filename)
