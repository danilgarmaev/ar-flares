"""Check the PowerPoint package and render companion previews / printable notes."""
from pathlib import Path
import json, zipfile
from xml.sax.saxutils import escape
from lxml import etree
from pptx import Presentation
import fitz
from PIL import Image, ImageDraw
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
path=ROOT/'ICANN_2026_Solar_Flare_Prediction_Engaging.pptx'
prs=Presentation(path)
notes=json.loads((ROOT/'speaker_notes.json').read_text())
MAIN_COUNT=sum(not n['backup'] for n in notes)
TOTAL=len(notes)
assert len(prs.slides)==TOTAL
assert all(s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip() for s in prs.slides)
assert all(s._element.get('show')!='0' for s in list(prs.slides)[:MAIN_COUNT])
assert all(s._element.get('show')=='0' for s in list(prs.slides)[MAIN_COUNT:])
bounds=[]
for i,s in enumerate(prs.slides,1):
    for sh in s.shapes:
        if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+300 or sh.top+sh.height>prs.slide_height+300:
            bounds.append((i,sh.name))
assert not bounds,bounds
layout=json.loads((ROOT/'layout_checks.json').read_text())
overflows=[r for r in layout if r['need_pt']>r['height_pt']+1]
assert not overflows,overflows
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    for n in z.namelist():
        if n.endswith('.xml') or n.endswith('.rels'):etree.fromstring(z.read(n))
    # All images are embedded; no file-system/network dependency in the deck.
    for n in z.namelist():
        if n.endswith('.rels'):
            root=etree.fromstring(z.read(n))
            assert not any('image' in el.get('Type','') and el.get('TargetMode')=='External' for el in root)

doc=fitz.open(ROOT/'ICANN_2026_Solar_Flare_Prediction_Engaging_preview.pdf')
assert len(doc)==TOTAL
thumbs=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
    name=ROOT/'preview'/f'slide_{i+1:02d}.png';pix.save(name)
    im=Image.open(name);im.thumbnail((640,360))
    th=Image.new('RGB',(660,396),'#dfe7ec');th.paste(im,(10,10))
    ImageDraw.Draw(th).text((12,375),f'SLIDE {i+1:02d}',fill='black');thumbs.append(th)
for j in range(0,len(thumbs),6):
    sheet=Image.new('RGB',(1320,1188),'white')
    for k,th in enumerate(thumbs[j:j+6]):sheet.paste(th,((k%2)*660,(k//2)*396))
    sheet.save(ROOT/'preview'/f'contact_{j//6+1}.png')

fonts=Path('/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/liberation-fonts')
pdfmetrics.registerFont(TTFont('Liberation',str(fonts/'LiberationSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('LiberationBold',str(fonts/'LiberationSans-Bold.ttf')))
pdfmetrics.registerFontFamily('Liberation',normal='Liberation',bold='LiberationBold',italic='Liberation',boldItalic='LiberationBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('TalkTitle',fontName='LiberationBold',fontSize=20,leading=25,textColor=HexColor('#142B3C'),spaceAfter=15))
styles.add(ParagraphStyle('TalkBody',fontName='Liberation',fontSize=12,leading=18,spaceAfter=12))
styles.add(ParagraphStyle('TalkMeta',fontName='Liberation',fontSize=10,leading=14,textColor=HexColor('#526777'),spaceAfter=12))
story=[];elapsed=0
notes=json.loads((ROOT/'speaker_notes.json').read_text())
for i,n in enumerate(notes):
    if i:story.append(PageBreak())
    if n['backup']:timing='Q&A backup — hidden during the main talk'
    else:
        end=elapsed+n['seconds'];timing=f'Target: {elapsed//60}:{elapsed%60:02d}–{end//60}:{end%60:02d} ({n["seconds"]} seconds)';elapsed=end
    story.append(Paragraph(f'SLIDE {n["slide"]:02d}  /  ICANN 2026',styles['TalkMeta']))
    story.append(Paragraph(escape(n['title']),styles['TalkTitle']))
    story.append(Paragraph(escape(timing),styles['TalkMeta']))
    story.append(Paragraph(escape(n['script']).replace('\n','<br/>'),styles['TalkBody']))
    for key,title in [('cue','Delivery / transition'),('say','How to say it'),('qa','If asked'),('source','Sources / verification')]:
        if n.get(key):story.append(Paragraph(f'<b>{title}</b><br/>'+escape(n[key]),styles['TalkMeta']))

def footer(c,d):
    c.setFont('Liberation',9);c.setFillColor(HexColor('#526777'))
    c.drawString(48,30,'Danil Garmaev · ICANN 2026 · Speaker notes')
    c.drawRightString(564,30,str(d.page))

notes_doc=SimpleDocTemplate(str(ROOT/'Speaker_Notes_Engaging.pdf'),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=48,title='ICANN 2026 — speaker notes',author='Danil Garmaev and Jean Meunier')
notes_doc.build(story,onFirstPage=footer,onLaterPages=footer)
notes_pages=len(fitz.open(ROOT/'Speaker_Notes_Engaging.pdf'))
report={
    'pptx_slides':TOTAL,'visible_main_slides':MAIN_COUNT,'hidden_backups':TOTAL-MAIN_COUNT,'slides_with_notes':TOTAL,
    'target_talk_seconds':elapsed,'main_script_words':sum(len(n['script'].split()) for n in notes if not n['backup']),
    'native_shape_count':sum(len(s.shapes) for s in prs.slides),
    'out_of_bounds_shapes':bounds,'text_height_overflows':overflows,
    'zip_and_xml_valid':True,'all_images_embedded':True,
    'companion_preview_pages':len(doc),'printable_notes_pages':notes_pages,
    'actual_powerpoint_or_libreoffice_render_performed':False,
    'render_method':'Companion vector PDF from shared drawing commands; Arial-compatible Liberation Sans metrics.',
    'pptx_size_bytes':path.stat().st_size,
}
(ROOT/'validation_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
