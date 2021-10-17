import docx2txt

def getTextWord(wordFileName):
    text = docx2txt.process(wordFileName)

    return text