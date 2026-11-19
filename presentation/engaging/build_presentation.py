"""Build the ICANN talk, editable PowerPoint, vector PDF preview and speaker guide.

Run with python-pptx, Pillow, reportlab and PyMuPDF installed.
All result values are transcribed from the accepted author proof, Tables 2–5.
Shapes and text are native PowerPoint objects; scientific images are real data.
"""
from pathlib import Path
import json, math, textwrap
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parent
ASSETS = OUT / 'assets'
W, H = 13.333333, 7.5
MAIN_COUNT = 14
NAVY='142B3C'; INK='172E3C'; TEAL='007F80'; ORANGE='C46534'
BG='F7F9FA'; WHITE='FFFFFF'; MUTED='526777'; GRID='D9E2E7'; LIGHT='E7F2F1'; PALE='F8EDE5'
FONT='Arial'
# Liberation Sans is metrically compatible with Arial used in PowerPoint.
FONTS=Path('/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/share/fonts/liberation-fonts')
pdfmetrics.registerFont(TTFont('DV', str(FONTS/'LiberationSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('DVB', str(FONTS/'LiberationSans-Bold.ttf')))
prs=Presentation(); prs.slide_width=Inches(W); prs.slide_height=Inches(H)
prs.core_properties.title='Solar Flare Prediction from Magnetograms: A Controlled Study of Input Design in Deep Learning'
prs.core_properties.author='Danil Garmaev and Jean Meunier'
prs.core_properties.subject='ICANN 2026 • Paper 750 • 15-minute oral presentation'
prs.core_properties.keywords='solar flare; magnetogram; MViT; input design; ICANN 2026'
pdf=canvas.Canvas(str(OUT/'ICANN_2026_Solar_Flare_Prediction_Engaging_preview.pdf'), pagesize=(W*72,H*72))
pdf.setTitle(prs.core_properties.title); pdf.setAuthor(prs.core_properties.author)
notes=[]; text_checks=[]; slides=[]

def rgb(c): return RGBColor.from_string(c)
def rect(s,x,y,w,h,fill=WHITE,line=None,radius=False):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
    if line:sh.line.color.rgb=rgb(line);sh.line.width=Pt(.7)
    else:sh.line.fill.background()
    pdf.setFillColor(HexColor('#'+fill));pdf.setStrokeColor(HexColor('#'+(line or fill)))
    pdf.setLineWidth(.7)
    pdf.rect(x*72,(H-y-h)*72,w*72,h*72,stroke=bool(line),fill=1)
    return sh

def line(s,x1,y1,x2,y2,color=GRID,width=1.2):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
    pdf.setStrokeColor(HexColor('#'+color));pdf.setLineWidth(width)
    pdf.line(x1*72,(H-y1)*72,x2*72,(H-y2)*72)
    return sh

def dot(s,x,y,r=.06,color=TEAL,outline=None):
    sh=s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x-r), Inches(y-r), Inches(2*r), Inches(2*r))
    sh.fill.solid();sh.fill.fore_color.rgb=rgb(color)
    if outline:sh.line.color.rgb=rgb(outline);sh.line.width=Pt(1)
    else:sh.line.fill.background()
    pdf.setFillColor(HexColor('#'+color));pdf.setStrokeColor(HexColor('#'+(outline or color)))
    pdf.circle(x*72,(H-y)*72,r*72,fill=1,stroke=bool(outline))
    return sh

def wrap(text,width,size,bold):
    name='DVB' if bold else 'DV'; out=[]
    for para in text.split('\n'):
        if not para:out.append('');continue
        words=para.split(); cur=''
        for word in words:
            test=(cur+' '+word).strip()
            if pdfmetrics.stringWidth(test,name,size)>width*72 and cur:
                out.append(cur);cur=word
            else:cur=test
        out.append(cur)
    return out

def txt(s,text,x,y,w,h,size=20,color=INK,bold=False,align='left',leading=1.16):
    # Explicit line breaks keep PowerPoint and the vector preview consistent.
    lines=wrap(text,w-.03,size,bold)
    rendered='\n'.join(lines)
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame;tf.clear();tf.word_wrap=False
    tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.TOP
    for i,t in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=t
        p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=Pt(size*leading)
        p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align]
    pdf.setFillColor(HexColor('#'+color));pdf.setFont('DVB' if bold else 'DV',size)
    for i,t in enumerate(lines):
        yy=(H-y)*72-size*.83-i*size*leading
        if align=='center':pdf.drawCentredString((x+w/2)*72,yy,t)
        elif align=='right':pdf.drawRightString((x+w)*72,yy,t)
        else:pdf.drawString(x*72,yy,t)
    need=(len(lines)-1)*size*leading+size*1.05
    text_checks.append({'slide':len(slides),'text':text[:100],'need_pt':round(need,1),'height_pt':round(h*72,1),'x':x,'y':y,'w':w,'h':h})
    return tb

def pic(s,path,x,y,w,h=None):
    path=Path(path)
    if h is None:
        im=Image.open(path);h=w*im.height/im.width
    s.shapes.add_picture(str(path),Inches(x),Inches(y),Inches(w),Inches(h))
    pdf.drawImage(str(path),x*72,(H-y-h)*72,w*72,h*72,mask='auto')

def start(title,section='INPUT DESIGN',dark=False,backup=False):
    s=prs.slides.add_slide(prs.slide_layouts[6]);slides.append(s)
    c=NAVY if dark else BG
    s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(c)
    pdf.setFillColor(HexColor('#'+c));pdf.rect(0,0,W*72,H*72,stroke=0,fill=1)
    if title:
        txt(s,section.upper(),.58,.32,11,.3,11,TEAL if not dark else '71D1C7',True)
        txt(s,title,.58,.91,12.2,1.0,30,WHITE if dark else INK,True)
    line(s,.58,7.06,12.75,7.06,'385061' if dark else GRID,.8)
    txt(s,'GARMAEV & MEUNIER  /  ICANN 2026',.58,7.19,6,.22,9,'A6BAC6' if dark else MUTED)
    label=f'B{len(slides)-MAIN_COUNT:02d}' if backup else f'{len(slides):02d} / {MAIN_COUNT}'
    txt(s,label,11.7,7.16,1.05,.25,10,'A6BAC6' if dark else MUTED,align='right')
    return s

def finish(s,title,seconds,script,cue='',source='',qa='',backup=False,say=''):
    n={'slide':len(slides),'title':title,'seconds':seconds,'script':textwrap.dedent(script).strip(),'cue':cue,'source':source,'qa':qa,'backup':backup,'say':say}
    notes.append(n)
    chunks=[('BACKUP — Q&A ONLY' if backup else f'TARGET TIME: {seconds} seconds'),n['script']]
    if cue:chunks+=['DELIVERY / TRANSITION\n'+cue]
    if say:chunks+=['HOW TO SAY IT\n'+say]
    if qa:chunks+=['IF ASKED\n'+qa]
    if source:chunks+=['SOURCES / VERIFICATION\n'+source]
    s.notes_slide.notes_text_frame.text='\n\n'.join(chunks)
    pdf.showPage()

def source(s,t):txt(s,t,.62,6.77,12,.23,9,MUTED)
def band(s,text,y=6.05,color=LIGHT,fg=TEAL):
    rect(s,.58,y,12.17,.5,color);txt(s,text,.78,y+.105,11.75,.3,16,fg,True)
def card(s,x,y,w,h,label,body,color=TEAL):
    rect(s,x,y,w,h,WHITE);rect(s,x,y,.045,h,color)
    txt(s,label,x+.22,y+.19,w-.42,.55,21,color,True)
    txt(s,body,x+.22,y+.91,w-.44,h-.99,18,INK)
def legend(s,x,y):
    dot(s,x+.07,y+.12,.055,TEAL);txt(s,'≥C',x+.22,y,0.65,.3,15,TEAL,True)
    dot(s,x+1.07,y+.12,.055,ORANGE);txt(s,'≥M',x+1.22,y,.65,.3,15,ORANGE,True)

def chart(s,x,y,w,h,labels,series,ymin=.45,ymax=.9,ticks=None,value_labels=False,xlabel='',ylabel='TSS',connect=True):
    """Editable native-shape point estimates and ±1 SD error bars."""
    if ticks is None:ticks=[.5,.6,.7,.8,.9]
    lx=x+.58;rx=x+w-.25;top=y+.3;bot=y+h-.62
    pos=[lx+(rx-lx)*(i+.5)/len(labels) for i in range(len(labels))]
    yy=lambda v:bot-(v-ymin)/(ymax-ymin)*(bot-top)
    txt(s,ylabel,x,y-.1,w,.27,12,MUTED,True)
    for t in ticks:
        if ymin<=t<=ymax:
            line(s,lx,yy(t),rx,yy(t),GRID,.65)
            txt(s,f'{t:.2f}',x,yy(t)-.09,.46,.22,11,MUTED,align='right')
    for i,l in enumerate(labels):txt(s,l,pos[i]-.62,bot+.16,1.24,.42,13,MUTED,align='center')
    for si,(name,vals,stds,col) in enumerate(series):
        off=(si-(len(series)-1)/2)*.11
        if connect:
            for i in range(len(vals)-1):line(s,pos[i]+off,yy(vals[i]),pos[i+1]+off,yy(vals[i+1]),col,2.0)
        for i,(v,sd) in enumerate(zip(vals,stds)):
            xx=pos[i]+off
            if sd:
                line(s,xx,yy(v-sd),xx,yy(v+sd),col,1.3)
                line(s,xx-.06,yy(v-sd),xx+.06,yy(v-sd),col,1.3)
                line(s,xx-.06,yy(v+sd),xx+.06,yy(v+sd),col,1.3)
            dot(s,xx,yy(v),.061,col)
            if value_labels:
                dy=-.33 if si==0 else .12
                txt(s,f'{v:.3f}',xx-.44,yy(v)+dy,.88,.28,13,col,True,align='center')
    if xlabel:txt(s,xlabel,x+.6,y+h-.02,w-.8,.3,12,MUTED,align='center')
    return pos,yy

def table(s,headers,rows,x=.68,y=2.0,width=12.0,row_h=.48,col_widths=None,size=15):
    if col_widths is None:col_widths=[width/len(headers)]*len(headers)
    rect(s,x,y,width,row_h,NAVY)
    xx=x
    for hd,cw in zip(headers,col_widths):txt(s,hd,xx+.12,y+.13,cw-.2,.3,size,WHITE,True);xx+=cw
    for i,row in enumerate(rows):
        yy=y+row_h*(i+1);rect(s,x,yy,width,row_h,WHITE if i%2==0 else 'EDF2F5')
        xx=x
        for j,(v,cw) in enumerate(zip(row,col_widths)):
            txt(s,str(v),xx+.12,yy+.13,cw-.2,row_h-.13,size,INK,j==0);xx+=cw

# Prepare faithful visual demonstrations using real data, never generated magnetograms.
raw=Image.open(ASSETS/'frame_15.png').convert('RGB')
for res in [224,112,56]:
    im=raw if res==224 else raw.resize((res,res),Image.Resampling.NEAREST).resize((224,224),Image.Resampling.BILINEAR)
    im.save(ASSETS/f'resolution_{res}.png')

PAPER='Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf.'
DATA='Boucheron et al., Scientific Data 10, 825 (2023), doi:10.1038/s41597-023-02628-8.'
MVIT='Fan et al., Multiscale Vision Transformers, ICCV 2021, pp. 6824–6835.'

# 01 — Title
s=start('',dark=True)
pic(s,ASSETS/'paper-000.png',7.93,1.37,4.82,4.82)
txt(s,'ICANN 2026  •  ORAL PRESENTATION  •  PAPER 750',.62,.57,12,.35,12,'71D1C7',True)
txt(s,'Solar Flare Prediction\nfrom Magnetograms',.62,1.46,7.08,1.65,34,WHITE,True)
txt(s,'A Controlled Study of Input Design\nin Deep Learning',.65,3.31,6.95,1.0,24,'BBD0DD')
line(s,.65,4.66,2.1,4.66,'71D1C7',3)
txt(s,'Danil Garmaev  ·  Jean Meunier',.65,5.03,7.0,.46,20,WHITE,True)
txt(s,'Université de Montréal\nComputer Science and Operations Research',.65,5.69,7.0,.8,16,'BBD0DD')
txt(s,'SDO/HMI magnetogram • reproduced from paper, Fig. 1',7.95,6.43,4.78,.26,9,'A6BAC6')
finish(s,'Solar Flare Prediction from Magnetograms',20,'Hello, I’m Danil Garmaev, a master’s student at Université de Montréal. With Jean Meunier, I studied a practical question in solar flare prediction: how much information does a model actually need? Our results point to useful choices about image detail and the way we sample the past.','Start slowly. Give the audience a moment to settle before moving to the background.',PAPER+' Figure 1; SDO/HMI imagery credited in the paper.', say='Garmaev and Meunier: use your usual pronunciation of your names. No need to read the full subtitle.')


# Background — solar flares and the class labels, before introducing magnetograms.
s=start('Solar flares: what they are and why we predict them','BACKGROUND')
pic(s,ASSETS/'solar_flare_nasa_noaa.jpg',.71,2.0,3.0,3.0)
txt(s,'A solar flare',.73,5.15,3.0,.4,20,INK,True)
txt(s,'NASA image via NOAA',.73,5.64,3.0,.28,11,MUTED)
txt(s,'A sudden release of magnetic energy',4.12,2.09,8.36,.48,25,INK,True)
txt(s,'Seen as a burst of radiation from the Sun’s atmosphere.',4.13,2.76,8.2,.65,20,INK)
txt(s,'Why predict it?',4.13,3.71,8.2,.45,23,TEAL,True)
txt(s,'Strong flares can disrupt radio communication.\nAdvance forecasts help people prepare.',4.13,4.34,8.2,.88,21,INK)
txt(s,'X-RAY STRENGTH',.74,6.16,1.95,.27,10,MUTED,True)
for j,(label,col) in enumerate([('A',MUTED),('B',MUTED),('C',TEAL),('M',ORANGE),('X',ORANGE)]):
    x=2.8+j*1.03
    rect(s,x,5.98,.78,.63,WHITE)
    txt(s,label,x+.04,6.07,.70,.44,25,col,True,align='center')
    if j<4:txt(s,'→',x+.81,6.12,.23,.3,16,MUTED)
txt(s,'M1 = 10 × C1\nX1 = 10 × M1',8.19,6.03,3.8,.65,17,INK)
source(s,'Sources: thesis §1.1 and Table 1.1; NOAA, Solar Flares (Radio Blackouts). Image courtesy of NASA, hosted by NOAA.')
finish(s,'Solar flares: what they are and why we predict them',90,'Before I get into the model, let me explain what we are trying to predict. A solar flare is a sudden release of magnetic energy in the Sun’s atmosphere. We observe it as a burst of radiation, like the bright region in this image. Strong flares can disrupt radio communication on the sunlit side of Earth. Forecasting them helps people prepare for those disruptions.\n\nThe letters along the bottom describe flare strength, measured using X-rays. The scale runs from A to X. Each step has a threshold ten times higher than the previous one: for example, M-one is ten times as intense as C-one in that measurement. We focus on two questions: will there be a C-class flare or stronger, and will there be an M-class flare or stronger? The second group contains the stronger, rarer events. In both cases, we want to predict what happens in the next twenty-four hours, before the flare occurs.','Point to the bright flare, then follow the letters from left to right. Explain the two tasks once; after this, “C-class or stronger” and “M-class or stronger” are enough.',source='Garmaev, master’s thesis, §1.1 and Table 1.1. NOAA: https://www.spaceweather.gov/phenomena/solar-flares-radio-blackouts . NASA: https://science.nasa.gov/sun/solar-storms-and-flares/ . Image: https://www.spaceweather.gov/sites/default/files/FlareImage.jpg',qa='Flare classes measure peak GOES soft X-ray flux, not total energy. M1 has ten times the peak flux of C1; comparing arbitrary M and C events is not a fixed factor of ten. A flare is a burst of radiation; a CME is an ejection of plasma and magnetic field. They can occur together, but the task here is flare prediction, not CME or geomagnetic-storm forecasting. Radiation reaches Earth as we observe the flare, so advance prediction differs from detecting a flare that has already occurred.',say='A, B, C, M, X: “ay, bee, see, em, ex.” ≥C: “C-class or stronger.” ≥M: “M-class or stronger.” M1 and C1: “M-one” and “C-one.”')

# 02 — Context
s=start('What the model sees','THE PREDICTION TASK')
pic(s,ASSETS/'paper-000.png',.66,2.0,3.5,3.5)
pic(s,ASSETS/'paper-001.png',4.52,2.52,2.62,2.62)
line(s,4.22,3.82,4.43,3.82,TEAL,2)
txt(s,'Full solar disk',.68,5.63,3.45,.4,17,INK,True)
txt(s,'Active-region cutout',4.45,5.63,2.88,.4,17,INK,True)
txt(s,'Black / white = opposite magnetic polarities',.7,6.15,6.6,.33,14,MUTED)
rect(s,7.65,2.05,5.02,3.99,WHITE)
txt(s,'INPUT',7.92,2.31,4.44,.3,12,TEAL,True)
txt(s,'One magnetogram\nor a sequence from\none region',7.92,2.81,4.42,1.15,21,INK,True)
line(s,7.94,4.09,12.34,4.09,GRID)
txt(s,'TWO PREDICTION TASKS',7.92,4.34,4.44,.3,12,TEAL,True)
txt(s,'C-class or stronger\nM-class or stronger\nin the next 24 hours',7.92,4.84,4.4,1.0,20,INK,True)
source(s,'Sources: paper §1–2 and Fig. 1; Boucheron et al. (2023); Scherrer et al. (2012).')
finish(s,'What the model sees',70,'To look for warning signs, we use images of the Sun’s magnetic field. These are called magnetograms. They are different from the flare image we just saw: black and white show magnetic fields pointing in opposite directions.\n\nWe focus on active regions, which are areas where the magnetic field is concentrated. The box on the full image shows one of these regions, and the smaller image shows the cutout we give to the model. The two views use different contrast settings, which is why they look different.\n\nThe model sees either one image or a series of images from the same region. It then estimates whether a flare above our chosen class will happen during the following day. So we are looking for signs before an event, rather than recognizing a flare that is already happening.','Point to the full disk and then the cutout. Make the distinction between the magnetic-field image and the earlier flare image explicit.',PAPER+' §1–2, Fig. 1. '+DATA, say='Magnetogram: “mag-NEE-toh-gram.” AR means “active region”; use the words in the talk. 24 h: “the next twenty-four hours” or “the following day.”')

# 03 — Four questions
s=start('Input design as a research question','THE CONTRIBUTION')
card(s,.65,2.02,5.87,1.7,'01  Spatial detail','Original and smaller images\nHow much detail do we need?')
card(s,6.8,2.02,5.87,1.7,'02  Disk position','Central and outer regions\nDoes position affect prediction?')
card(s,.65,4.01,5.87,1.7,'03  Temporal span','16 images over different periods\nHow far back should we look?',ORANGE)
card(s,6.8,4.01,5.87,1.7,'04  Frame count','Fewer images over the same period\nHow many images do we need?',ORANGE)
band(s,'We change the input while keeping the model family and training setup consistent.',6.12)
source(s,'Source: paper §4. Comparisons retain the architecture family and a shared training protocol.')
finish(s,'Input design as a research question',50,'Our contribution is a controlled study of input design. We ask four questions: how much spatial detail matters, how the viewing position affects prediction, how far back the model should look, and how many observations it needs within that period.\n\nThe last two are especially useful to separate. A sequence can contain many images yet cover very little time. Or it can contain just a few observations spread over a longer period.\n\nWe study these choices within the same model family and a shared training procedure. This gives us evidence for designing the input alongside the architecture. I’ll start with the spatial results, then show why the temporal findings are particularly interesting.','Treat this as a short roadmap. Do not read all settings from the four boxes.',PAPER+' §4.1–4.7.', say='224 × 224: “two twenty-four by two twenty-four pixels,” if needed. “Image size” is usually enough here.')

# 04 — Data and evaluation
s=start('Data and evaluation','DATA & EVALUATION')
txt(s,'SDO/HMI  •  2010–2018  •  224 × 224 pixels',.7,1.93,11.8,.45,22,INK,True)
for x,w,lab,num in [(0.7,5.66,'TRAIN','759,357'),(6.59,2.92,'VALIDATION','95,933'),(9.74,2.92,'TEST','94,757')]:
    rect(s,x,2.63,w,1.15,WHITE);txt(s,lab,x+.18,2.79,w-.34,.25,11,MUTED,True);txt(s,num,x+.18,3.15,w-.34,.43,24,TEAL,True)
txt(s,'Different active regions in training, validation and test sets',.76,4.02,11.8,.4,20,INK,True)
txt(s,'POSITIVE FRACTION IN TRAINING',.75,4.72,5.8,.32,12,MUTED,True)
for y,lab,frac,col in [(5.28,'≥C',.197,TEAL),(5.89,'≥M',.031,ORANGE)]:
    txt(s,lab,.76,y-.03,.7,.36,19,col,True);rect(s,1.52,y,3.55,.24,GRID);rect(s,1.52,y,3.55*frac,.24,col)
    txt(s,f'{frac*100:.1f}%',5.3,y-.04,1.0,.4,20,col,True)
rect(s,7.05,4.61,5.59,2.0,LIGHT)
txt(s,'Primary metric: TSS',7.3,4.82,5.07,.4,21,TEAL,True)
txt(s,'TSS = sensitivity − false-positive rate',7.3,5.34,5.07,.36,18,INK,True)
txt(s,'= sensitivity + specificity − 1',7.3,5.79,5.07,.32,16,INK)
txt(s,'1 = perfect; 0 = no skill.',7.3,6.23,5.07,.27,14,MUTED)
source(s,'Sources: paper §2–3. Counts describe the image benchmark; eligible sequence counts differ. TSS: Bloomfield et al. (2012).')
finish(s,'Data and evaluation',65,'The images come from the Solar Dynamics Observatory. Each active region belongs to only one of the training, validation or test sets, so we test on regions unseen during training.\n\nStrong flares are rare: only about three percent of M-task training examples are positive. Always predicting no flare could look accurate without being useful.\n\nWe therefore use the True Skill Statistic, or TSS: sensitivity minus the false-positive rate. Sensitivity is the fraction of positive examples we correctly detect. The false-positive rate is the fraction of negative examples that trigger a warning. TSS rewards detection while penalizing false alarms. One is perfect; zero means no skill by this measure. We choose the warning threshold using validation data.','Leave the exact dataset counts on screen. Point to the orange bar, then the first equation. Explain that equation in words; the equivalent specificity form is there for readers familiar with it.',PAPER+' §2–3; dataset image counts and training positive fractions.', 'TSS = sensitivity + specificity − 1 = TP/(TP+FN) − FP/(FP+TN). Sensitivity (recall or true-positive rate) = TP/(TP+FN). Specificity (true-negative rate) = TN/(TN+FP) = 1 − false-positive rate. Positive means a flare at or above the task threshold within the next 24 hours. TSS ranges from −1 to 1; negative values indicate a hit rate below the false-positive rate. A trivial all-negative classifier has TSS 0. False-positive rate, FP/(FP+TN), differs from the fraction of warnings that are false, FP/(TP+FP). Validation threshold selection targets TSS with recall ≥0.5 and precision ≥0.15 constraints.', say='TSS: “tee ess ess.” First equation: “TSS equals sensitivity minus the false-positive rate.” Equivalent form, only if asked: “sensitivity plus specificity minus one.” Specificity: “spess-ih-FISS-ih-tee.” 3.1%: “about three percent.”')

# 05 — Pipeline
s=start('A common framework for input comparisons','METHOD')
for j,i in enumerate([0,5,10,15]):pic(s,ASSETS/f'frame_{i:02d}.png',.74+j*.25,2.36+j*.2,1.65,1.65)
txt(s,'Image / sequence',.73,4.85,2.8,.38,18,INK,True)
txt(s,'Same active region',.73,5.35,2.8,.35,15,MUTED)
for x,lab,body in [(3.65,'MViT backbone','Looks at small patterns\nand larger structures'),(8.02,'Binary prediction','A prediction for the\nfollowing day')]:
    rect(s,x,2.76,3.83,2.02,WHITE);txt(s,lab,x+.2,3.0,3.43,.75,23,TEAL,True)
    txt(s,body,x+.2,3.86,3.43,.72,17,INK)
line(s,3.0,3.74,3.5,3.74,TEAL,2);line(s,7.56,3.74,7.9,3.74,TEAL,2)
txt(s,'Pretrained image / video models',3.68,5.17,4.08,.8,17,MUTED)
txt(s,'10 random seeds per setting',8.05,5.17,4.04,.8,17,MUTED)
band(s,'We repeat training 10 times to see how much the results vary.')
source(s,'Sources: paper §4 and Table 1; Fan et al. (2021). Real input frames: repository AR 1339, 2011-11-04 to 2011-11-06 (TAI).')
finish(s,'A common framework for input comparisons',50,'We use MViT, a Multiscale Vision Transformer. It combines information at different spatial scales, from local patterns to the broader structure of an active region. Its image and sequence versions let us study both inputs within one model family.\n\nWe fine-tune pretrained models on magnetograms and repeat each setting ten times. The error bars show variation between those training runs, so we can distinguish a consistent pattern from a favorable run.\n\nThis framework lets us focus on the scientific question: which parts of the available information contribute to forecasting skill? The detailed training settings are in the backups.','Spend more time on what the model receives than on the optimizer. Explain the error bars once here.',PAPER+' §4.1–4.7, Table 1. '+MVIT,'There is no claim of a new MViT architecture. The controlled factors are input choices within the shared model family and fine-tuning protocol.', say='MViT: “em-vit,” followed once by “Multiscale Vision Transformer.” “Ten training runs” is clearer than repeatedly saying “ten random seeds.”')

# 06 — Resolution
s=start('Forecasting with less spatial detail','01 / SPATIAL DETAIL')
for i,r in enumerate([224,112,56]):
    pic(s,ASSETS/f'resolution_{r}.png',.72+i*1.67,2.25,1.46,1.46)
    txt(s,f'{r} × {r}',.7+i*1.67,3.9,1.5,.38,16,INK,True,align='center')
txt(s,'Same input size.\nLess image detail.',.75,4.67,4.93,.83,20,INK,True)
txt(s,'Useful signal survives coarser detail.\nSpeed is not measured here.',.75,5.61,4.93,.65,16,MUTED)
legend(s,6.02,1.95)
chart(s,5.85,2.58,6.61,3.37,['224','112','56'],[('C',[.534,.539,.522],[.026,.061,.064],TEAL),('M',[.647,.618,.618],[.046,.050,.062],ORANGE)],ymin=.43,ymax=.73,ticks=[.45,.50,.55,.60,.65,.70],xlabel='Spatial resolution before resizing')
source(s,'Source: paper Table 2 and §4.3. Points: mean TSS; bars: ±1 SD over 10 seeds. No significant pairwise TSS differences reported.')
finish(s,'Forecasting with less spatial detail',55,'First, how much spatial detail does the prediction need? We deliberately make the magnetograms coarser, then resize them back to the model’s expected dimensions.\n\nThe striking feature is how little the TSS changes. For C-class or stronger flares, the trend is nearly flat. For the stronger M-class task, full resolution has the highest average, but the differences are not statistically clear.\n\nThis suggests that useful predictive information survives substantial loss of fine detail. It gives us a reason to investigate more compact image representations. It does not establish that fine structure is irrelevant, or that the model runs faster: the final input dimensions are unchanged.','Describe the mostly flat trend. There is no need to read any TSS values on this slide.',PAPER+' §4.3, §5.1, Table 2. Visual pipeline applied to repository frame_15 (AR 1339).', say='56 × 56: “fifty-six by fifty-six.” 112 × 112: “one twelve by one twelve.” “Original, half-size and quarter-size images” is also fine.')

# 07 — Longitude
s=start('Viewing geometry affects the comparison','02 / SOLAR LONGITUDE')
pic(s,ASSETS/'latlon_60.png',.75,2.18,2.38,2.46)
pic(s,ASSETS/'latlon_30.png',3.50,2.18,2.38,2.46)
txt(s,'Full range: ±60°',.71,4.87,2.57,.4,19,INK,True,align='center')
txt(s,'Central region: ±30°',3.28,4.87,2.89,.4,19,INK,True,align='center')
txt(s,'Blue dots: active-region centers\nRegions near the edge are viewed at an angle.',.81,5.55,5.43,.83,17,MUTED)
txt(s,'One model, two viewing regions',6.58,1.98,5.72,.46,21,INK,True)
legend(s,6.70,2.58)
chart(s,6.33,3.19,6.15,2.48,['Central','Outer'],[('C',[.863,.843],[.015,.015],TEAL),('M',[.916,.887],[.013,.025],ORANGE)],ymin=.8,ymax=.95,ticks=[.8,.85,.9,.95],value_labels=False,ylabel='ROC-AUC  •  higher is better')
rect(s,6.62,5.88,5.80,.63,LIGHT)
txt(s,'Central-only training: mixed results.',6.80,6.03,5.42,.34,16,TEAL,True)
source(s,'Sources: thesis Fig. 2.2 (restored latlon_60 / latlon_30 figures); paper Table 3. Mean ROC-AUC ±1 SD over 10 runs.')

finish(s,'Viewing geometry affects the comparison',70,'A magnetogram also depends on where we observe the region. Near the center of the visible Sun, we see it more directly. Farther out, projection changes its apparent shape and the magnetic field we measure along our line of sight.\n\nThe disk diagrams show the full and central ranges. On the right, the same model achieves a higher average ROC-AUC on central regions for both tasks. ROC-AUC measures how well it separates flaring from non-flaring examples across thresholds.\n\nBut training only on central regions gives mixed results. A cleaner viewing geometry does not automatically make a better training set. Removing outer regions also removes data and changes the population. The implication is to evaluate where a model works well before deciding which observations to exclude.','Explain the restored disk diagrams before discussing the chart. Blue dots are region centers, not flare events. Keep numerical specialization results in backup B03.',PAPER+' §4.4, §5.2, Table 3; thesis Fig. 2.2; restored latlon_30.png and latlon_60.png; repository A2 evaluation records.', 'Preparation audit found that saved A2 evaluations use Best_threshold from test evaluation, while the paper describes validation-selected thresholds. This deck therefore omits longitude TSS/F1 and associated p-values. Confirm or reevaluate A2 with validation-selected thresholds before making those claims; no new training or reevaluation was performed for this deck.', say='ROC-AUC: “rock A-U-C.” Longitude: “LON-jih-tood.” ±30°: “within thirty degrees of the center.” Do not read the absolute-value notation aloud.')

# 08 — Temporal design
s=start('Temporal coverage and sampling density','03–04 / TEMPORAL DESIGN')
txt(s,'A. Temporal coverage: 16 images over more time',.74,1.94,11.85,.48,23,INK,True)
xx=3.7;end=12.2;scale=(end-xx)/45
for y,span,cad in [(2.82,15,60),(3.38,30,120),(3.94,45,180)]:
    txt(s,f'{cad} min → {span} h',.77,y-.17,2.5,.37,19,TEAL,True)
    startx=end-span*scale;line(s,startx,y,end,y,TEAL,1.8)
    for i in range(16):dot(s,startx+(end-startx)*i/15,y,.041,TEAL)
txt(s,'B. Sampling density: fewer images over the same period',.74,4.65,11.85,.48,23,INK,True)
txt(s,'4 images',.77,5.59,2.5,.37,19,ORANGE,True)
line(s,xx,5.77,end,5.77,ORANGE,1.8)
for hours in [45,30,15,0]:
    xp=end-hours*scale;dot(s,xp,5.77,.072,ORANGE);txt(s,f'−{hours} h' if hours else 'Now',xp-.49,6.05,.98,.3,15,ORANGE,True,align='center')
txt(s,'Make the prediction',10.91,2.28,1.83,.27,10,MUTED,align='center')
source(s,'Sources: paper §4.5–4.6. Temporal span = (T − 1) × cadence for uniformly spaced frames. Forecast horizon remains the next 24 h.')
finish(s,'Temporal coverage and sampling density',65,'The temporal experiments separate two choices that are easy to confuse: the period we observe and the number of images we collect within it.\n\nAt the top, each sequence has sixteen images and ends when we make the prediction. Increasing the time between images extends our view into the past. The longest setting covers forty-five hours, almost two days.\n\nAt the bottom, we keep that same period but reduce the number of observations. We can then ask whether the useful information requires dense sampling, or whether a few widely spaced images remain informative.\n\nKeep the two time windows in mind: we observe the past and predict the following twenty-four hours. We are not predicting forty-five hours ahead.','Trace the right-aligned endpoints first. Use your pointer to show “more time covered” versus “fewer dots.”',PAPER+' §4.5–4.6; repository frame-count configs: T4 offsets [-225,-150,-75,0] in 12-minute steps.', 'The implemented T8 subset uses lookbacks [45,39,33,27,18,12,6,0] h; it is approximately spaced on the available grid, not eight perfectly equal intervals.', say='45 h: “forty-five hours,” or “almost two days” when explaining the idea. Cadence means “time between images”; prefer that phrase.')

# 09 — Cadence
s=start('Longer windows improve C-class prediction','03 / TEMPORAL SPAN')
legend(s,.84,1.96)
chart(s,.69,2.64,8.2,3.76,['15 h\n60 min','30 h\n120 min','45 h\n180 min'],[('C',[.674,.742,.789],[.021,.028,.021],TEAL),('M',[.626,.671,.706],[.062,.039,.107],ORANGE)],ymin=.5,ymax=.85,ticks=[.5,.6,.7,.8],value_labels=True)
rect(s,9.21,2.34,3.42,3.9,WHITE)
txt(s,'+0.115',9.49,2.75,2.86,.69,36,TEAL,True)
txt(s,'TSS improvement\nfor C-class or stronger',9.49,3.58,2.86,.95,20,INK,True)
txt(s,'Significant improvement\nover 10 training runs',9.49,4.83,2.84,.83,17,TEAL,True)
txt(s,'M-class or stronger:\na less certain result',9.49,5.69,2.85,.61,13,MUTED)
source(s,'Source: paper Table 4. Mean ±1 SD, 10 seeds. ≥C paired tests: p < 0.001, p = 0.0013, p < 0.001. Eligible cohorts vary with cadence.')
finish(s,'Longer windows improve C-class prediction',70,'This is the clearest improvement in the study. With sixteen images in every sequence, extending the observation period improves C-class-or-stronger prediction: TSS rises from about zero point six seven to zero point seven nine.\n\nThe improvement is statistically significant across the repeated runs. The ranking metrics improve as well, so the finding is supported by more than one way of evaluating the predictions. For M-class or stronger, the mean moves in the same direction, but the uncertainty is larger.\n\nWhy does this matter? The model receives the same number of images, but they cover more of the region’s evolution. That makes temporal coverage a promising design choice.\n\nThere is one important qualification: longer sequences change which examples are available. Comparing exactly the same prediction examples is the next step toward isolating the effect of the longer window.','This is the one result where reading two rounded values helps. Then point to the trend and the larger orange error bars. Leave p-values to questions.',PAPER+' §5.3, Table 4. Repository aggregation of fixed_TSS reproduced all six means and sample SDs to 3 decimals.', 'Cadence and span change together in this experiment. The cohort-count backup shows sample attrition. These results support, but do not prove a purely causal temporal-span explanation.', say='0.674 → 0.789: say “about zero point six seven to zero point seven nine.” These are TSS scores, not percentages of correct predictions.')

# 10 — Frames
s=start('Four images remain competitive','04 / FRAME COUNT')
for j,(i,t) in enumerate([(0,'−45 h'),(5,'−30 h'),(10,'−15 h'),(15,'Now')]):
    pic(s,ASSETS/f'frame_{i:02d}.png',.76+j*1.31,2.44,1.15,1.15)
    txt(s,t,.73+j*1.31,3.8,1.22,.32,15,INK,True,align='center')
txt(s,'Competitive TSS with four images',.79,4.57,5.25,.85,24,TEAL,True)
txt(s,'First and last images: 45 hours apart\nThree quarters fewer input images',.79,5.57,5.14,.85,17,INK)
legend(s,6.54,1.99)
chart(s,6.16,2.7,6.35,3.58,['16','8','4'],[('C',[.789,.802,.805],[.021,.033,.025],TEAL),('M',[.706,.751,.752],[.107,.075,.069],ORANGE)],ymin=.55,ymax=.9,ticks=[.6,.7,.8,.9],value_labels=True,xlabel='Images over the same 45-hour period')
source(s,'Source: paper Table 5. Mean ±1 SD over 10 seeds. TSS differences were not significant. Real example: AR 1339, 4–6 Nov 2011 (TAI).')
finish(s,'Four images remain competitive',70,'Now comes the complementary result. Once the images cover almost two days, reducing the sequence from sixteen observations to four leaves TSS competitive for both tasks.\n\nThe four magnetograms on the left show what that input looks like: an early observation, two intermediate views, and the latest image. The model still has access to change over the full period.\n\nThe differences in TSS are not statistically significant, so this is evidence for a useful sparse setting, not proof that four images are optimal or equivalent to sixteen.\n\nTogether with the previous result, this suggests that covering the relevant period may matter more than sampling it densely. Four images also mean three quarters fewer inputs, which motivates measuring practical savings in speed and memory. But TSS is only one part of the picture; the stronger-flare task reveals a trade-off.','Point to the four real observations. Say “the same period” or “almost two days,” rather than “preserving a 45-hour history.”',PAPER+' §5.4, Table 5. AR1339 frames copied unchanged from data/wds_out/train/shard-000015.tar through shard-000017.tar.', '75% = (16−4)/16. Lack of statistical significance is not an equivalence or non-inferiority test. No measured runtime or memory savings are claimed.', say='75%: “seventy-five percent” or, more naturally here, “three quarters fewer images.” Do not read all six plotted TSS values.')

# 11 — Metric tradeoff
s=start('Rare events: interpreting PR-AUC','READING THE RESULTS')
txt(s,'M-class or stronger  •  same 45-hour period',.76,1.94,11.83,.45,21,INK,True)
pos,yy=chart(s,.72,2.91,6.45,3.16,['16 images','8 images','4 images'],[('PR-AUC',[.393,.389,.359],[0,0,0],TEAL)],ymin=0,ymax=1,ticks=[0,.2,.4,.6,.8,1],value_labels=True,ylabel='PR-AUC  •  1 is a perfect ranking')
baseline=73/2702
line(s,1.30,yy(baseline),6.92,yy(baseline),ORANGE,1.6)
txt(s,'Random ranking ≈ 0.03',1.46,yy(baseline)-.40,4.43,.3,15,ORANGE,True)
rect(s,7.58,2.76,5.06,3.38,WHITE)
txt(s,'About 3 in 100',7.83,3.05,4.51,.59,29,ORANGE,True)
txt(s,'test examples are followed\nby an M-class flare or stronger.',7.83,3.85,4.48,.82,20,INK)
txt(s,'0.36–0.39 is well above the\nrandom-ranking reference.',7.83,5.03,4.46,.80,20,TEAL,True)
band(s,'PR-AUC is not accuracy. It summarizes the balance between finding flares and reliable warnings.',6.18)
source(s,'Sources: paper Table 5; thesis §1.2. Reference = 73 positives / 2,702 evaluated examples. Plotted scores are means; SDs in backup B05.')

finish(s,'Rare events: interpreting PR-AUC',95,'To understand that trade-off, we need to read this metric on the right scale. PR-AUC summarizes the precision–recall curve: how reliable our warnings are as we try to detect more of the flares, across different thresholds.\n\nA score around zero point three six is not thirty-six percent accuracy. In this test set, only about three in a hundred examples are followed by an M-class flare or stronger. The conventional random-ranking reference is therefore about zero point zero three, near the bottom of the plot. Our scores sit well above that reference, showing that the model has learned useful predictive information.\n\nThat does not mean the forecasting problem is solved. With fewer images, mean PR-AUC decreases even as mean TSS increases. The apparent advantage depends on the metric we choose.\n\nFor an application, we would need to decide how to balance missed flares and false warnings, then evaluate a suitable operating threshold. The useful conclusion here is that sparse inputs remain promising, with a trade-off that TSS alone would hide.','Point to the random-ranking reference before the model points. Do not say “thirteen times more accurate,” and do not treat PR-AUC as precision at a single threshold.',PAPER+' §5.4, Table 5; thesis §1.2; pr_baseline_audit.json. The exact random-ranking reference is 73/2702, about 0.027, in all 30 M-task frame-count runs.', 'TSS is not alert precision. Calibration, deployment prevalence, costs of missed events and false alarms, and a pre-specified operating policy would matter for operational deployment.', say='PR-AUC: “P-R A-U-C.” On first use: “the area under the precision–recall curve.” 0.36: “zero point three six.” 0.03: “zero point zero three.”')

# 12 — Scope
s=start('From input findings to stronger benchmarks','SCOPE & NEXT EXPERIMENTS')
for y,n,obs,nxt in [
    (2.12,'01','Match examples and benchmark models','Use the same examples for input settings and strong published baselines'),
    (3.37,'02','Measure the practical savings','Check speed and memory use when the model receives fewer images'),
    (4.62,'03','Test on other data','Check whether the results hold for other regions and time periods')]:
    txt(s,n,.75,y,.65,.55,26,TEAL,True)
    txt(s,obs,1.67,y+.01,10.87,.45,23,INK,True)
    txt(s,nxt,1.68,y+.59,10.75,.5,19,MUTED)
band(s,'Compare input designs and model architectures under the same evaluation protocol.',6.15)
source(s,'Source: paper §6; repository sequence-count audit. Seed variation does not measure generalization uncertainty across solar cycles.')
finish(s,'From input findings to stronger benchmarks',55,'These findings suggest a focused next stage. First, match the regions and prediction times across input settings. Include a single-image baseline using the last frame of each sequence, and strong published models under that same protocol.\n\nSecond, measure speed and memory use. Reducing the number of input images creates an opportunity for practical savings, but those savings need measurement.\n\nThird, test other time periods to see how well the findings generalize.\n\nThose experiments would connect the input-design results to a stronger performance benchmark. I have a backup comparison with published methods for the discussion; the current study’s main contribution is the evidence about the input choices themselves.','Present three concrete next steps. Keep additional limitations in the backup notes.',PAPER+' §6; presentation/pr_baseline_audit.json.', 'A stronger static baseline uses the final frame of the same sequence-derived examples with matched labels, splits and training budget. A solar-cycle holdout probes a different generalization question from the current AR-disjoint split.', say='“The same examples” is clearer than “matched cohorts.” “Other time periods” is enough unless someone asks specifically about solar cycles.')

# 13 — Conclusion and questions
s=start('What this study contributes','TAKEAWAYS',dark=True)
for y,num,head,body in [
    (2.23,'01','Longer observation windows improved skill','A significant TSS improvement for C-class flares or stronger.'),
    (3.58,'02','Four observations remained competitive','Sparse sequences retained useful information over almost two days.'),
    (4.93,'03','Coarser images retained predictive information','Limited TSS changes after reducing spatial detail; geometry effects were mixed.')]:
    txt(s,num,.72,y,.75,.5,27,'71D1C7',True)
    txt(s,head,1.73,y,10.8,.6,24,WHITE,True)
    txt(s,body,1.74,y+.64,10.72,.49,17,'BBD0DD')
txt(s,'Thank you  •  Questions',.74,6.49,7,.34,19,'71D1C7',True)
txt(s,'danil.garmaev@umontreal.ca',8.45,6.54,4.22,.32,14,'BBD0DD',align='right')
finish(s,'What this study contributes',40,'Three findings stand out. Extending the observation period improved C-class-or-stronger prediction. Four images remained competitive when they covered that longer period. And reducing spatial detail had a limited effect on TSS in these experiments.\n\nTogether, these results give us a practical direction: preserve informative changes over time, then test how much spatial and temporal detail the model really needs. Input design deserves to be evaluated alongside model design. Thank you.','Finish here at roughly 14 minutes 25 seconds. Keep this slide visible for questions; use the hidden backups only when needed.',PAPER+' §7; Tables 2–5.', say='No numbers need to be read on the closing slide.')

# B01 — Training
s=start('Training and evaluation details','BACKUP B01',backup=True)
table(s,['Setting','Image model','Sequence model'],[
    ['Backbone','Pretrained 2D MViT','Pretrained MViT'],
    ['Input','1 × 224 × 224','T × 1 × 224 × 224'],
    ['Optimizer / decay','AdamW / 0.01','AdamW / 0.01'],
    ['Learning rates','Backbone 1e-6; head 1e-4','Backbone 1e-6; head 1e-4'],
    ['Schedule / warmup','Cosine / 2 epochs','Cosine / 3 epochs'],
    ['Batch / max epochs','32 / 50','8 / 50'],
    ['Loss / repetitions','Weighted BCE / 10 seeds','Weighted BCE / 10 seeds'],
],y=1.95,row_h=.47,col_widths=[3.2,4.4,4.4],size=15)
txt(s,'Threshold: maximize validation TSS, targeting recall ≥0.5 and precision ≥0.15.\nAugmentation: H/V flips, rotations ≤±30°, polarity inversion, integer noise in [−5, 5].',.77,6.0,11.8,.68,14,INK)
source(s,'Source: paper Table 1 and §3–4. Repository configs include negative subsampling; benchmark counts are not effective training counts.')
finish(s,'B01 — Training and evaluation details',0,'Use this slide only for a training-protocol question. The learning rates differ between the pretrained backbone and classification head. Geometric transforms and polarity inversion are consistent across a sequence; noise can vary across frames. Early stopping and checkpoint selection use validation information. Repository configs also include negative subsampling, so do not interpret the full image-benchmark counts as the number of effective training examples for every run.',source=PAPER+' §3, §4.2, Table 1; repository saved config.json files.',qa='The code can fall back to unconstrained TSS if no threshold satisfies the constraints. The constraints describe the selection target, not a guarantee about test precision or recall.',backup=True)

# B02 — Resolution complete
s=start('Resolution study: all reported metrics','BACKUP B02',backup=True)
table(s,['Task / pixels','TSS','ROC-AUC','PR-AUC','F1'],[
    ['≥C / 224','.534 ± .026','.833 ± .010','.641 ± .024','.620 ± .013'],
    ['≥C / 112','.539 ± .061','.834 ± .031','.640 ± .048','.613 ± .041'],
    ['≥C / 56','.522 ± .064','.829 ± .032','.626 ± .049','.605 ± .043'],
    ['≥M / 224','.647 ± .046','.910 ± .007','.391 ± .047','.309 ± .028'],
    ['≥M / 112','.618 ± .050','.902 ± .013','.373 ± .055','.296 ± .048'],
    ['≥M / 56','.618 ± .062','.898 ± .018','.324 ± .055','.296 ± .032'],
],y=2.08,row_h=.51,col_widths=[2.36,2.41,2.41,2.41,2.41],size=14)
band(s,'Reduced spatial detail remains useful; lack of significance does not establish equivalence.',6.08)
source(s,'Source: paper Table 2. All values are mean ± SD across 10 seeds.')
finish(s,'B02 — Resolution study: all reported metrics',0,'The full-resolution M-task setting has the highest means on all four metrics, and its PR-AUC advantage over the coarsest setting is more noticeable than the TSS difference. Do not generalize a non-significant TSS test to every metric. The resolution operation changes the information available before resizing; the final network dimensions remain fixed.',source=PAPER+' Table 2.',backup=True)

# B03 — Longitude complete
s=start('Longitude: threshold-free ranking metrics','BACKUP B03',backup=True)
table(s,['Task','Training','Test','ROC-AUC','PR-AUC'],[
    ['≥C','Full','Full','.833 ± .010','.641 ± .024'],
    ['≥C','Full','Inner','.863 ± .015','.637 ± .038'],
    ['≥C','Full','Outer','.843 ± .015','.726 ± .023'],
    ['≥C','Central','Inner','.852 ± .010','.623 ± .025'],
    ['≥C','Central','Outer','.830 ± .007','.712 ± .015'],
    ['≥M','Full','Full','.910 ± .007','.391 ± .047'],
    ['≥M','Full','Inner','.916 ± .013','.455 ± .060'],
    ['≥M','Full','Outer','.887 ± .025','.359 ± .049'],
    ['≥M','Central','Inner','.920 ± .010','.470 ± .066'],
    ['≥M','Central','Outer','.907 ± .008','.316 ± .045'],
],y=1.87,row_h=.39,col_widths=[1.4,2.25,2.25,3.05,3.05],size=13)
txt(s,'Inner: |lon| ≤30°; outer: 30°<|lon|≤60°. PR-AUC comparisons also depend on the positive fraction in each population.',.74,6.4,11.95,.35,12,MUTED)
source(s,'Source: paper Table 3. Values: mean ± SD across 10 seeds. Descriptive ranking comparisons; no significance claim.')
finish(s,'B03 — Longitude: threshold-free ranking metrics',0,'Compare inner versus outer only within the same training setup. Compare specialization using the same central test subset. The central-only model is also evaluated on the outer bucket, but those numbers do not establish that central-only training is a general improvement. PR-AUC depends on the positive fraction of the evaluated population. Longitude TSS and F1 are intentionally omitted pending resolution of the threshold-provenance discrepancy found in the repository. The analysis memo has concrete evidence and the recommended reevaluation.',source=PAPER+' Table 3; A2 JSON evaluation records; Analysis_and_Verification.md.',backup=True)

# B04 — Cadence complete
s=start('Cadence study: 16 frames, three temporal spans','BACKUP B04',backup=True)
table(s,['Task / cadence','TSS','ROC-AUC','PR-AUC','F1'],[
    ['≥C / 60 min','.674 ± .021','.910 ± .015','.766 ± .033','.677 ± .015'],
    ['≥C / 120 min','.742 ± .028','.934 ± .011','.810 ± .033','.718 ± .022'],
    ['≥C / 180 min','.789 ± .021','.958 ± .005','.855 ± .029','.736 ± .025'],
    ['≥M / 60 min','.626 ± .062','.896 ± .013','.405 ± .070','.277 ± .039'],
    ['≥M / 120 min','.671 ± .039','.914 ± .010','.366 ± .076','.289 ± .040'],
    ['≥M / 180 min','.706 ± .107','.947 ± .013','.393 ± .065','.362 ± .078'],
],y=2.08,row_h=.51,col_widths=[2.36,2.41,2.41,2.41,2.41],size=14)
txt(s,'≥C paired TSS tests: 120 vs. 60, p < .001; 180 vs. 120, p = .0013; 180 vs. 60, p < .001.\n≥M pairwise TSS differences were not significant.',.76,6.0,11.8,.64,14,INK)
source(s,'Source: paper Table 4. Mean ± SD over 10 seeds. Repository fixed-threshold TSS aggregates reproduce these values.')
finish(s,'B04 — Cadence study: full results',0,'For C-or-stronger, all four mean metrics improve across the three tested spans. For M-or-stronger, TSS and ROC-AUC peak at 180 minutes, but PR-AUC peaks at 60 minutes. This reinforces the metric-dependent interpretation. The significant tests reported here concern TSS only. Do not imply that all changes in AUC, PR-AUC or F1 were tested or found significant.',source=PAPER+' Table 4, §5.3.',backup=True)

# B05 — Frames complete
s=start('Frame-count study: fixed 45-hour endpoints','BACKUP B05',backup=True)
table(s,['Task / frames','TSS','ROC-AUC','PR-AUC','F1'],[
    ['≥C / 16','.789 ± .021','.958 ± .005','.855 ± .029','.736 ± .025'],
    ['≥C / 8','.802 ± .033','.962 ± .006','.871 ± .017','.737 ± .037'],
    ['≥C / 4','.805 ± .025','.960 ± .005','.862 ± .015','.741 ± .028'],
    ['≥M / 16','.706 ± .107','.947 ± .013','.393 ± .065','.362 ± .078'],
    ['≥M / 8','.751 ± .075','.952 ± .010','.389 ± .081','.352 ± .059'],
    ['≥M / 4','.752 ± .069','.945 ± .012','.359 ± .061','.327 ± .057'],
],y=2.08,row_h=.51,col_widths=[2.36,2.41,2.41,2.41,2.41],size=14)
txt(s,'Repository lookbacks: T4 = [45, 30, 15, 0] h; T8 = [45, 39, 33, 27, 18, 12, 6, 0] h.\nT16 samples every 3 h. No frame padding, repetition or temporal interpolation.',.76,6.0,11.8,.64,14,INK)
source(s,'Sources: paper Table 5; repository saved frame-count configs. Mean ± SD over 10 seeds. TSS differences were not significant.')
finish(s,'B05 — Frame-count study: full results',0,'This table supports a viable sparse regime rather than a unique optimum. For C-or-stronger, TSS means are close. For M-or-stronger, the larger TSS means with sparse inputs coexist with lower mean PR-AUC and F1. The eight-frame implementation selects an approximately spaced subset from the three-hour grid, with a nine-hour gap in the middle. Do not claim mathematically uniform spacing for that configuration. The endpoints remain exactly forty-five hours apart.',source=PAPER+' Table 5; saved config.json seq_offsets fields and scripts/run_mvit_framecount_unique_5seed_a100.slurm.',backup=True)

# B06 — Repository analysis
s=start('Longer sequences leave fewer test examples','BACKUP B06 / REPOSITORY CHECK',backup=True)
table(s,['Time between images','Period covered','Test examples','M-or-stronger examples'],[
    ['60 min','15 h','14,551','494'],
    ['120 min','30 h','5,475','165'],
    ['180 min','45 h','2,702','73'],
],y=2.2,row_h=.6,col_widths=[3.1,2.1,2.7,4.1],size=16)
txt(s,'Why this matters',.78,5.0,4.4,.5,24,TEAL,True)
txt(s,'The time-window comparison also changes the examples being evaluated.\nNext step: compare the same prediction examples across all settings.',.78,5.72,11.73,.77,19,INK)
source(s,'Source: saved confusion counts in the 30 M-task cadence runs. Counts agree across 10 seeds per setting; see pr_baseline_audit.json.')
finish(s,'B06 — Test example counts',0,'These counts come directly from the evaluated test predictions in the saved run metrics. Longer sequences leave fewer usable test examples. The forty-five-hour setting has 73 M-or-stronger positives out of 2,702 examples, which gives the approximately 0.03 random-ranking reference used on the PR-AUC slide. The four-, eight- and sixteen-frame settings all have these same class counts across their ten runs. Equal counts alone do not prove that all example identities match. These are correlated sequence examples, not counts of independent flare events.',source='presentation/pr_baseline_audit.json, using the run paths saved in temporal_results_audit.json. Earlier cached sequence_size_counts differed and are not used here.',backup=True)

# B07 — Q&A
s=start('Questions to be ready for','BACKUP B07',backup=True)
for y,q,a in [
    (2.02,'What is new?','Controlled input-design comparisons within a common MViT framework.'),
    (3.08,'Why not claim a best frame count?','Differences are uncertain, and preferred settings change across metrics.'),
    (4.14,'Does this establish operational readiness?','No prospective validation, calibration study or deployment cost model yet.'),
    (5.20,'What is the most useful next experiment?','Match the eligible samples across cadences and the static baseline.')]:
    txt(s,q,.76,y,11.9,.43,22,TEAL,True);txt(s,a,.77,y+.51,11.75,.42,17,INK)
source(s,'Discussion guidance grounded in paper §6; see embedded notes for more methodological questions and answers.')
finish(s,'B07 — Q&A preparation',0,'Answer the direct question first, then one supporting fact. These prompts are a rehearsal aid; normally keep the final takeaway slide visible and navigate directly to the relevant evidence table.',source=PAPER+' §6.',qa='Why MViT? It supports multiscale image and temporal representations in a common family; architecture novelty is not the claim. Why ten seeds? To quantify optimization variability; this does not replace AR-level or temporal uncertainty analysis. Why TSS? It balances hit rate and false-positive rate under class imbalance, but it is not sufficient for an operational policy. Is 180 minutes optimal? It is the strongest mean TSS/AUC among the paper’s three 16-frame settings, not a global optimum. Did lower resolution save compute? No measured backbone speedup: inputs were returned to 224×224. Are four frames equivalent to sixteen? No formal equivalence test; they are competitive in these results. Is the threshold tuned on test? The reported main TSS uses a validation-selected threshold; repository metrics also contain other threshold variants, which must not be substituted. Is a 3D model proven better than a 2D model? No: use a matched last-frame baseline before making that claim.',backup=True)

# B08 — References
s=start('Selected references and visual provenance','BACKUP B08',backup=True)
refs=[
    ('This study','Garmaev, D.; Meunier, J. Solar Flare Prediction from Magnetograms: A Controlled Study of Input Design in Deep Learning. Accepted ICANN 2026 paper 750.'),
    ('Dataset','Boucheron et al. Solar active region magnetogram image dataset for studies of space weather. Scientific Data 10, 825 (2023). doi:10.1038/s41597-023-02628-8'),
    ('Skill metric','Bloomfield et al. Toward reliable benchmarking of solar flare forecasting methods. ApJL 747, L41 (2012). doi:10.1088/2041-8205/747/2/L41'),
    ('Instrument','Scherrer et al. The Helioseismic and Magnetic Imager (HMI) investigation for the Solar Dynamics Observatory. Solar Physics 275, 207–227 (2012). doi:10.1007/s11207-011-9834-2'),
    ('Architecture','Fan et al. Multiscale Vision Transformers. ICCV (2021), pp. 6824–6835.'),
]
for i,(lab,body) in enumerate(refs):
    yy=1.96+i*.85;txt(s,lab,.75,yy,1.72,.4,15,TEAL,True);txt(s,body,2.58,yy,10.04,.74,13,INK)
txt(s,'Background: Garmaev, master’s thesis (2026), §1.1–1.2; NOAA, Solar Flares (Radio Blackouts).\nVisuals: NASA via NOAA; thesis Fig. 2.2; paper Fig. 1; real repository AR 1339 images.',.76,6.36,11.8,.44,11,MUTED)
finish(s,'B08 — References and visual provenance',0,'These references are copied from the accepted paper. The main charts are newly drawn for readability using the paper’s numeric results. The magnetograms are actual data, not generated scientific images. The cohort analysis is explicitly identified as supplementary repository material.',source=PAPER+' references 1–3 and 17; Garmaev_Danil_2026_memoire.pdf §1.1–1.2 and Fig. 2.2; https://www.spaceweather.gov/phenomena/solar-flares-radio-blackouts ; https://science.nasa.gov/sun/solar-storms-and-flares/ ; presentation/assets/frame_provenance.json.',backup=True)

# B09 — Published performance context, not a matched leaderboard.
s=start('How does this compare with published methods?','BACKUP B09 / SOTA DISCUSSION',backup=True)
txt(s,'Selected ≥M / next-24-hour results  •  reported TSS, mean ± SD',.76,1.91,11.9,.42,19,INK,True)
table(s,['Study / model','Input and evaluation','Reported TSS'],[
    ['Riggi et al. / SigLIP2','HMI images; downsampled Boucheron split','0.671 ± 0.027'],
    ['Riggi et al. / VideoMAE','HMI sequences; downsampled Boucheron split','0.613 ± 0.050'],
    ['Alatoom & Nikolaou / ViT + temporal model','SDOBenchmark; multi-wavelength; AR aggregation','0.806 ± 0.041'],
    ['This study / image MViT','Boucheron image benchmark; 224 × 224','0.647 ± 0.046'],
    ['This study / 4-image MViT','Boucheron sequence subset; 45-hour window','0.752 ± 0.069'],
],y=2.57,row_h=.55,col_widths=[4.05,5.58,2.37],size=13)
txt(s,'Different samples, inputs and evaluation units prevent a direct ranking.',.78,6.02,11.8,.43,20,TEAL,True)
txt(s,'Our contribution: evidence about input design. Establishing a performance lead requires matched baselines.',.78,6.49,11.8,.28,12,INK)
source(s,'Sources: Riggi et al., arXiv:2510.23400v1, Table 6 (default split, tuned threshold); Alatoom & Nikolaou (2026), Table 5; our Tables 2, 5.')
finish(s,'B09 — Published performance / SOTA context',0,
    'We report useful forecasting skill, but this study does not establish a state-of-the-art ranking. For the M-class-or-stronger task, our four-image setting reaches a mean TSS of about zero point seven five. Published scores provide context, but these methods were not evaluated on the same examples with the same inputs. The contribution I can defend directly is the input-design comparison within our framework. A matched evaluation against strong baselines is the next step.',
    cue='Answer the SOTA question directly, then show this table if a numerical comparison is useful. Do not read every row. This is slide 23 / B09; B10 gives the methodological answer.',
    source='Riggi et al., Solar flare forecasting with foundational transformer models across image, video, and time-series modalities, https://arxiv.org/html/2510.23400v1 , Table 6: default split, tau-star column, five runs. Alatoom & Nikolaou (2026), Multi-Wavelength Transformer-Based 24-Hour Solar Flare Forecasting at the Active-Region Level, https://doi.org/10.1029/2026SW005010 , Table 5: six seeds. Own results: accepted paper Tables 2 and 5, ten seeds. Sources checked 14 September 2026.',
    qa='Selected relevant reports, not an exhaustive SOTA leaderboard. Riggi values come from the explicitly identified v1 preprint table; its later full-encoder VideoMAE experiment reports 0.636 ± 0.046. Alatoom uses multiple SDO wavelengths and maximum probability aggregation per active region, a different evaluation unit from our sequence examples. Do not interpret the numerical gaps as measured improvements or infer significance from these standard deviations. Our single-image and sequence rows also use different eligible examples. The C-task TSS of 0.805 must not be compared with these M-task scores.',
    backup=True,say='SOTA: “state of the art.” For our M-task result, say “a TSS of about zero point seven five,” not “seventy-five percent accuracy.”')

# B10 — Concise answer and a concrete fair-benchmark design.
s=start('What would establish a fair performance comparison?','BACKUP B10 / SOTA FOLLOW-UP',backup=True)
for y,lab,body in [
    (2.04,'Match the forecasting task','Same flare threshold, forecast horizon, examples and active-region splits.'),
    (3.16,'Match the evaluation','Same threshold-selection policy, metrics and unit of prediction.'),
    (4.28,'Use strong reference models','Last-frame baseline, sequence baseline and published models on those same examples.'),
]:
    txt(s,lab,.77,y,11.8,.43,22,TEAL,True)
    txt(s,body,.78,y+.53,11.75,.57,18,INK)
band(s,'The defensible contribution today: a systematic study of which input choices matter.',6.13)
source(s,'Proposed follow-up, not an experiment reported in the paper. Published score context and sources: backup B09.')
finish(s,'B10 — Answering the SOTA follow-up',0,
    'The strongest test would use the same prediction examples for every model, including a baseline that sees only the last image. We would keep the flare class, forecast horizon, data split and evaluation policy fixed. That would let us ask whether our input choices transfer to other architectures and whether they improve performance under a shared benchmark. What this study already provides is a set of testable findings: longer temporal coverage helped the C task, sparse observations remained competitive, and substantial spatial coarsening caused limited changes in TSS.',
    cue='Use only if the question continues beyond the performance table. Lead with the answer, then the experiment that would resolve it. This is slide 24 / B10.',
    source=PAPER+' §4–6; proposed benchmark design informed by the documented protocol differences in B09.',
    qa='If asked “Why was there no SOTA benchmark?”: The study was designed around input comparisons within MViT; a shared-data comparison with other model families was outside the reported experiments. If asked “What is novel without a record score?”: Separating temporal coverage from observation count produces practical hypotheses for input design, supported by repeated runs. Do not claim a new architecture, a global optimum, or that these choices necessarily transfer to every model.',backup=True)

# Hide appendix from a normal oral-presentation slideshow, but retain navigation.
for s in slides[MAIN_COUNT:]:s._element.set('show','0')
prs.save(OUT/'ICANN_2026_Solar_Flare_Prediction_Engaging.pptx')
pdf.save()
(OUT/'speaker_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
(OUT/'layout_checks.json').write_text(json.dumps(text_checks,ensure_ascii=False,indent=2))
md=['# ICANN 2026 speaker guide — Engaging version','', 'Paper 750 · Danil Garmaev and Jean Meunier','',
    '14 main slides · target 14 minutes 25 seconds · 35 seconds of buffer · 5 minutes for questions.',
    '', 'Engaging version: stronger contribution framing, with the earlier clear version preserved in the parent directory. Dedicated SOTA discussion: B09 (slide 23) and B10 (slide 24). The same notes are embedded in the PowerPoint Notes pane. Backup slides are hidden from normal slideshow progression; access them through PowerPoint’s slide navigator.', '']
elapsed=0
for n in notes:
    if not n['backup']:
        end=elapsed+n['seconds'];timing=f"{elapsed//60}:{elapsed%60:02d}–{end//60}:{end%60:02d}";elapsed=end
    else:timing='Q&A backup'
    md += [f"## {n['slide']:02d}. {n['title']} ({timing})",'',n['script'],'']
    for k,title in [('cue','Delivery'),('say','How to say it'),('qa','If asked'),('source','Source')]:
        if n[k]:md += [f'**{title}:** {n[k]}','']
(OUT/'Speaker_Notes_Engaging.md').write_text('\n'.join(md))
overflow=[r for r in text_checks if r['need_pt']>r['height_pt']+1]
print(json.dumps({'slides':len(slides),'main_slides':MAIN_COUNT,'timing_seconds':elapsed,'overflow_count':len(overflow),'overflows':overflow},ensure_ascii=False,indent=2))
