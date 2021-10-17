from PyPDF4 import PdfFileReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import BytesIO


def getTextPDF(pdfFileName, password=''):
    pdfFile = open(pdfFileName, 'rb')
    readPDF = PdfFileReader(pdfFile)

    # decrypt password protected file, if any
    if (password != ''):
        readPDF.decrypt(password)

    # read text from the file
    text = []
    for i in range(0, readPDF.getNumPages()):
        text.append(readPDF.getPage(i).extractText())

    # return the single string object by joining the contents of all string
    # objects inside the list with a new line
    return '\n'.join(text)


def getTextPDFMiner(pdfFileName):
    manager = PDFResourceManager()
    retstr = BytesIO()
    layout = LAParams(all_texts=True)
    device = TextConverter(manager, retstr, laparams=layout)
    filepath = open(pdfFileName, 'rb')
    interpreter = PDFPageInterpreter(manager, device)

    for page in PDFPage.get_pages(filepath, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    filepath.close()
    device.close()
    retstr.close()

    return text
