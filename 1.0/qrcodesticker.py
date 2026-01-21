from PyPDF2 import PdfWriter, PdfReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab_qrcode import QRCodeImage

output = PdfWriter()

for i in range(111,131):
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    qr = QRCodeImage(str(i), size=30 * mm)
    qr.drawOn(can, 500, 710)
    can.drawString(500, 700, str(i))
    can.save()

    #move to the beginning of the StringIO buffer
    packet.seek(0)

    # create a new PDF with Reportlab
    new_pdf = PdfReader(packet)
    # read your existing PDF
    existing_pdf = PdfReader(open("db9.pdf", "rb"))
    # add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.pages[0]
    page.merge_page(new_pdf.pages[0])
    output.add_page(page)
# finally, write "output" to a real file
output_stream = open("../../AppData/Roaming/JetBrains/PyCharm2023.1/scratches/destination111-130.pdf", "wb")
output.write(output_stream)
output_stream.close()


    # doc.save()