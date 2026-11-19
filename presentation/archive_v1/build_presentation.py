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
pdf=canvas.Canvas(str(OUT/'ICANN_2026_Solar_Flare_Prediction_preview.pdf'), pagesize=(W*72,H*72))
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
    label=f'B{len(slides)-13:02d}' if backup else f'{len(slides):02d} / 13'
    txt(s,label,11.7,7.16,1.05,.25,10,'A6BAC6' if dark else MUTED,align='right')
    return s

def finish(s,title,seconds,script,cue='',source='',qa='',backup=False):
    n={'slide':len(slides),'title':title,'seconds':seconds,'script':textwrap.dedent(script).strip(),'cue':cue,'source':source,'qa':qa,'backup':backup}
    notes.append(n)
    chunks=[('BACKUP — Q&A ONLY' if backup else f'TARGET TIME: {seconds} seconds'),n['script']]
    if cue:chunks+=['DELIVERY / TRANSITION\n'+cue]
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
finish(s,'Solar Flare Prediction from Magnetograms',30,"""
Good [morning/afternoon]. I’m Danil Garmaev from Université de Montréal. This work with Jean Meunier asks a practical question about solar flare prediction: what information should we give a deep learning model? We study spatial detail, position on the solar disk, and temporal sampling. Our clearest finding is that covering a longer history of an active region can be more useful than giving the model many closely spaced observations.
""",'Pause after the central question. The bracketed greeting is for rehearsal only.',PAPER+' Figure 1; SDO/HMI imagery credited in the paper.')

# 02 — Context
s=start('Forecast the next 24 hours from magnetic structure','THE PREDICTION TASK')
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
txt(s,'TARGET',7.92,4.34,4.44,.3,12,TEAL,True)
txt(s,'A ≥C or ≥M flare\nin the next 24 hours?',7.92,4.84,4.4,1.0,22,INK,True)
source(s,'Sources: paper §1–2 and Fig. 1; Boucheron et al. (2023); Scherrer et al. (2012).')
finish(s,'Forecast the next 24 hours from magnetic structure',60,"""
Solar activity can affect satellites, communications and electrical infrastructure, so forecasting is useful before an event happens. The input here is a magnetogram: an image of the line-of-sight magnetic field. Black and white show opposite polarities; this is not an ordinary photograph of the Sun. The box identifies an active region, and the cutout is the image used by the learning pipeline. The full disk and cutout have different display scaling, so their contrasts should not be compared quantitatively. We ask two binary questions: will a C-class-or-stronger flare occur in the next 24 hours, and will an M-class-or-stronger flare occur? These are separate tasks. We can give the model a single observation or several observations of the same region leading up to prediction time.
""",'Point to the boxed region, then the cutout. Establish the forecast horizon before discussing the lookback window.',PAPER+' §1–2, Fig. 1. '+DATA)

# 03 — Four questions
s=start('Four input choices, one experimental framework','THE CONTRIBUTION')
card(s,.65,2.02,5.87,1.7,'01  Spatial detail','224² → 112² → 56²\nHow much image detail is useful?')
card(s,6.8,2.02,5.87,1.7,'02  Disk position','Central vs. outer longitudes\nWhere does the model generalize?')
card(s,.65,4.01,5.87,1.7,'03  Temporal span','16 frames over 15, 30 or 45 hours\nHow far back should we look?',ORANGE)
card(s,6.8,4.01,5.87,1.7,'04  Frame count','4, 8 or 16 frames over 45 hours\nHow densely should we observe?',ORANGE)
band(s,'Contribution: practical input-design evidence within an MViT fine-tuning framework.',6.12)
source(s,'Source: paper §4. Comparisons retain the architecture family and a shared training protocol.')
finish(s,'Four input choices, one experimental framework',55,"""
Our contribution is a controlled study of input design. First, we reduce the spatial detail in an active-region image. Second, we examine performance at different solar longitudes and test central-region specialization. Third, we hold the number of frames at sixteen and increase the temporal span. Fourth, we keep the span at forty-five hours and reduce the number of frames. We use the MViT architecture family throughout, with a common fine-tuning protocol. That lets us focus on practical decisions about the observations. The controls differ by experiment: for example, changing the required temporal history can also change which sequences are eligible. I will return to that limitation when interpreting the temporal results.
""",'Use this as the roadmap. Do not spend time enumerating alternative architectures.',PAPER+' §4.1–4.7.')

# 04 — Data and evaluation
s=start('Separate active regions; evaluate beyond accuracy','DATA & EVALUATION')
txt(s,'SDO/HMI  •  2010–2018  •  224 × 224 pixels',.7,1.93,11.8,.45,22,INK,True)
for x,w,lab,num in [(0.7,5.66,'TRAIN','759,357'),(6.59,2.92,'VALIDATION','95,933'),(9.74,2.92,'TEST','94,757')]:
    rect(s,x,2.63,w,1.15,WHITE);txt(s,lab,x+.18,2.79,w-.34,.25,11,MUTED,True);txt(s,num,x+.18,3.15,w-.34,.43,24,TEAL,True)
txt(s,'Disjoint active-region identities across the three splits',.76,4.02,11.8,.4,20,INK,True)
txt(s,'POSITIVE FRACTION IN TRAINING',.75,4.72,5.8,.32,12,MUTED,True)
for y,lab,frac,col in [(5.28,'≥C',.197,TEAL),(5.89,'≥M',.031,ORANGE)]:
    txt(s,lab,.76,y-.03,.7,.36,19,col,True);rect(s,1.52,y,3.55,.24,GRID);rect(s,1.52,y,3.55*frac,.24,col)
    txt(s,f'{frac*100:.1f}%',5.3,y-.04,1.0,.4,20,col,True)
rect(s,7.05,4.61,5.59,1.85,LIGHT)
txt(s,'Primary metric: TSS',7.3,4.82,5.07,.4,21,TEAL,True)
txt(s,'Hit rate − false-alarm rate\nThreshold chosen on validation data',7.3,5.4,5.07,.8,18,INK)
source(s,'Sources: paper §2–3. Counts describe the image benchmark; eligible sequence counts differ. TSS: Bloomfield et al. (2012).')
finish(s,'Separate active regions; evaluate beyond accuracy',70,"""
We use the reduced-resolution Boucheron dataset, built from SDO/HMI observations between 2010 and 2018. These are the image-benchmark counts. The predefined split separates active-region identities: a region in training does not also appear in validation or test. This reduces active-region leakage, although it is not a prospective solar-cycle holdout. The two tasks are imbalanced. About twenty percent of training images are positive for the C-or-stronger task, but only about three percent for M-or-stronger. Accuracy alone would therefore be misleading. Our primary score is the True Skill Statistic: the hit rate minus the false-positive rate. We choose the operating threshold on validation data and then evaluate on test data. We also report ROC-AUC, PR-AUC and F1. The sequence experiments use eligible temporal samples, so their counts are smaller than the image counts shown here.
""",'Explain the denominator distinction: false-positive rate is not the fraction of issued alerts that are false.',PAPER+' §2–3; dataset image counts and training positive fractions.', 'TSS = TP/(TP+FN) − FP/(FP+TN). A trivial all-negative classifier has TSS 0. Validation threshold selection targets TSS with recall ≥0.5 and precision ≥0.15 constraints.')

# 05 — Pipeline
s=start('Keep learning consistent while changing the inputs','METHOD')
for j,i in enumerate([0,5,10,15]):pic(s,ASSETS/f'frame_{i:02d}.png',.74+j*.25,2.36+j*.2,1.65,1.65)
txt(s,'Image / sequence',.73,4.85,2.8,.38,18,INK,True)
txt(s,'Same active region',.73,5.35,2.8,.35,15,MUTED)
for x,lab,body in [(3.65,'MViT backbone','Multiscale spatial features\n+ time for sequence inputs'),(8.02,'Binary prediction','Weighted binary\ncross-entropy')]:
    rect(s,x,2.76,3.83,2.02,WHITE);txt(s,lab,x+.2,3.0,3.43,.75,23,TEAL,True)
    txt(s,body,x+.2,3.86,3.43,.72,17,INK)
line(s,3.0,3.74,3.5,3.74,TEAL,2);line(s,7.56,3.74,7.9,3.74,TEAL,2)
txt(s,'Pretrained image / video models',3.68,5.17,4.08,.8,17,MUTED)
txt(s,'10 random seeds per setting',8.05,5.17,4.04,.8,17,MUTED)
band(s,'Shared fine-tuning protocol • validation-based checkpoint selection • 10 random seeds')
source(s,'Sources: paper §4 and Table 1; Fan et al. (2021). Real input frames: repository AR 1339, 2011-11-04 to 2011-11-06 (TAI).')
finish(s,'Keep learning consistent while changing the inputs',75,"""
MViT provides a multiscale feature hierarchy, which is useful here because the magnetic information may occur at both local and active-region scales. We use a two-dimensional MViT for images and an MViT sequence model for temporal inputs. Both start from pretrained models. Fine-tuning uses AdamW, weighted binary cross-entropy, and a cosine learning-rate schedule. The backbone learning rate is smaller than the classification head learning rate. We use augmentation during training, including flips, limited rotations and polarity inversion. For a sequence, geometric transforms and polarity inversion are shared across frames so that augmentation does not create inconsistent motion. The maximum budget is fifty epochs, with validation-based model selection and early stopping. Each reported configuration has ten random seeds. The plotted error bars summarize standard deviation across those seeds; they represent training variability, not uncertainty across independent solar cycles.
""",'The architecture schematic is conceptual; it does not specify exact layer counts. Save hyperparameters for backup B01.',PAPER+' §4.1–4.7, Table 1. '+MVIT,'There is no claim of a new MViT architecture. The controlled factors are input choices within the shared model family and fine-tuning protocol.')

# 06 — Resolution
s=start('Much of the predictive signal survives coarser images','01 / SPATIAL DETAIL')
for i,r in enumerate([224,112,56]):
    pic(s,ASSETS/f'resolution_{r}.png',.72+i*1.67,2.25,1.46,1.46)
    txt(s,f'{r} × {r}',.7+i*1.67,3.9,1.5,.38,16,INK,True,align='center')
txt(s,'All inputs return to 224 × 224\nbefore entering the backbone.',.75,4.67,4.93,.83,20,INK,True)
txt(s,'A test of retained spatial information.\nNo measured inference speedup.',.75,5.61,4.93,.65,16,MUTED)
legend(s,6.02,1.95)
chart(s,5.85,2.58,6.61,3.37,['224','112','56'],[('C',[.534,.539,.522],[.026,.061,.064],TEAL),('M',[.647,.618,.618],[.046,.050,.062],ORANGE)],ymin=.43,ymax=.73,ticks=[.45,.50,.55,.60,.65,.70],xlabel='Spatial resolution before resizing')
source(s,'Source: paper Table 2 and §4.3. Points: mean TSS; bars: ±1 SD over 10 seeds. No significant pairwise TSS differences reported.')
finish(s,'Much of the predictive signal survives coarser images',65,"""
Here we reduce the available spatial information while keeping the backbone input dimensions unchanged. Images are downsampled with nearest-neighbor interpolation and then returned to 224 by 224 with bilinear interpolation. These real magnetograms illustrate the operation. For C-or-stronger prediction, mean TSS stays close to 0.53 across all three resolutions. For M-or-stronger, full resolution has the highest mean TSS, but the lower-resolution settings remain fairly close. The paper reports no significant pairwise TSS differences. This supports a cautious conclusion: coarse images preserve much of the information used by this model. It does not prove that small-scale magnetic structure is unimportant, and it is not a runtime benchmark. Because every image is resized back to the same backbone size, we have not demonstrated an inference speedup from this spatial experiment.
""",'Point to image detail, then the modest TSS changes. Use “retains much of the signal,” not “equivalent.”',PAPER+' §4.3, §5.1, Table 2. Visual pipeline applied to repository frame_15 (AR 1339).')

# 07 — Longitude
s=start('Disk position matters; specialization is task-dependent','02 / SOLAR LONGITUDE')
txt(s,'Same full-training model',.75,2.06,6.05,.48,23,INK,True)
txt(s,'Inner: |longitude| ≤ 30°   •   Outer: 30–60°',.75,2.7,6.02,.46,16,MUTED)
legend(s,.84,3.28)
chart(s,.71,3.76,6.0,2.4,['Inner','Outer'],[('C',[.863,.843],[.015,.015],TEAL),('M',[.916,.887],[.013,.025],ORANGE)],ymin=.8,ymax=.95,ticks=[.8,.85,.9,.95],value_labels=False,ylabel='ROC-AUC')
rect(s,7.18,2.05,5.39,3.92,WHITE)
txt(s,'Train only on central regions?',7.43,2.33,4.89,.83,22,INK,True)
txt(s,'On the same central test subset:',7.43,3.27,4.87,.46,16,MUTED)
txt(s,'≥C    0.863 → 0.852',7.43,3.99,4.7,.45,24,TEAL,True)
txt(s,'Lower mean ROC-AUC',7.43,4.49,4.75,.37,15,MUTED)
txt(s,'≥M    0.916 → 0.920',7.43,5.03,4.7,.45,24,ORANGE,True)
txt(s,'Small increase in mean ROC-AUC',7.43,5.52,4.75,.37,15,MUTED)
band(s,'Threshold-free ranking: evaluate disk position explicitly; specialization is not uniformly beneficial.',6.16)
source(s,'Source: paper Table 3. Points: mean ROC-AUC; bars: ±1 SD over 10 seeds. Descriptive comparisons; no AUC significance claim.')
finish(s,'Disk position matters; specialization is task-dependent',70,"""
We separate two questions here and use ROC-AUC, which does not require an operating threshold. On the left, the same model trained over the full longitude range is evaluated on inner and outer regions. The inner bucket has higher mean ROC-AUC for both tasks: 0.863 versus 0.843 for C-or-stronger, and 0.916 versus 0.887 for M-or-stronger. This is consistent with position-dependent reliability and projection effects, although different region populations can also contribute. On the right, we compare full-range training with central-only training while keeping the central test subset fixed. C-task mean AUC decreases from 0.863 to 0.852, while M-task mean AUC increases slightly from 0.916 to 0.920. These are descriptive comparisons, not AUC significance claims. The useful conclusion is to evaluate disk position explicitly: central-only specialization is not uniformly beneficial.
""",'Separate “where we test” from “where we train.” Use the threshold-free AUC evidence shown here. See Analysis_and_Verification.md before discussing longitude TSS significance.',PAPER+' §4.4, §5.2, Table 3; repository A2 evaluation records.', 'Preparation audit found that saved A2 evaluations use Best_threshold from test evaluation, while the paper describes validation-selected thresholds. This deck therefore omits longitude TSS/F1 and associated p-values. Confirm or reevaluate A2 with validation-selected thresholds before making those claims; no new training or reevaluation was performed for this deck.')

# 08 — Temporal design
s=start('Separate how far back we look from how often we sample','03–04 / TEMPORAL DESIGN')
txt(s,'A. Fix 16 frames; extend the history',.74,1.94,11.85,.48,23,INK,True)
xx=3.7;end=12.2;scale=(end-xx)/45
for y,span,cad in [(2.82,15,60),(3.38,30,120),(3.94,45,180)]:
    txt(s,f'{cad} min → {span} h',.77,y-.17,2.5,.37,19,TEAL,True)
    startx=end-span*scale;line(s,startx,y,end,y,TEAL,1.8)
    for i in range(16):dot(s,startx+(end-startx)*i/15,y,.041,TEAL)
txt(s,'B. Fix 45 hours; reduce the observations',.74,4.65,11.85,.48,23,INK,True)
txt(s,'4 frames',.77,5.59,2.5,.37,19,ORANGE,True)
line(s,xx,5.77,end,5.77,ORANGE,1.8)
for hours in [45,30,15,0]:
    xp=end-hours*scale;dot(s,xp,5.77,.072,ORANGE);txt(s,f'−{hours} h' if hours else 'Now',xp-.49,6.05,.98,.3,15,ORANGE,True,align='center')
txt(s,'Prediction time',11.28,2.28,1.42,.27,11,MUTED,align='center')
source(s,'Sources: paper §4.5–4.6. Temporal span = (T − 1) × cadence for uniformly spaced frames. Forecast horizon remains the next 24 h.')
finish(s,'Separate how far back we look from how often we sample',70,"""
This distinction is central to the talk. In the first experiment, every input has sixteen frames. A sixty-minute cadence covers fifteen hours; a hundred-and-twenty-minute cadence covers thirty hours; and a hundred-and-eighty-minute cadence covers forty-five hours. The span is the number of intervals, fifteen, multiplied by the interval length. All inputs end at prediction time. Increasing the interval here therefore also extends the available history. In the second experiment, we hold the forty-five-hour window fixed and reduce the observations. The four-frame input samples forty-five, thirty and fifteen hours before prediction, plus the current frame. This asks whether densely observing that window is necessary. The eight-frame configuration also spans the same endpoints using a subset of the available grid. In both experiments the label still concerns the following twenty-four hours: lookback and forecast horizon are different quantities.
""",'Trace the right-aligned endpoints, then the four orange observations. Keep the two time windows distinct.',PAPER+' §4.5–4.6; repository frame-count configs: T4 offsets [-225,-150,-75,0] in 12-minute steps.', 'The implemented T8 subset uses lookbacks [45,39,33,27,18,12,6,0] h; it is approximately spaced on the available grid, not eight perfectly equal intervals.')

# 09 — Cadence
s=start('Longer history gives the clearest improvement for ≥C','03 / TEMPORAL SPAN')
legend(s,.84,1.96)
chart(s,.69,2.64,8.2,3.76,['15 h\n60 min','30 h\n120 min','45 h\n180 min'],[('C',[.674,.742,.789],[.021,.028,.021],TEAL),('M',[.626,.671,.706],[.062,.039,.107],ORANGE)],ymin=.5,ymax=.85,ticks=[.5,.6,.7,.8],value_labels=True)
rect(s,9.21,2.34,3.42,3.9,WHITE)
txt(s,'+0.115',9.49,2.75,2.86,.69,36,TEAL,True)
txt(s,'absolute TSS gain\nfor ≥C: 15 → 45 h',9.49,3.58,2.86,.95,20,INK,True)
txt(s,'All ≥C pairwise\nTSS tests significant',9.49,4.83,2.84,.83,17,TEAL,True)
txt(s,'≥M: similar mean trend;\nnot significant',9.49,5.69,2.85,.61,13,MUTED)
source(s,'Source: paper Table 4. Mean ±1 SD, 10 seeds. ≥C paired tests: p < 0.001, p = 0.0013, p < 0.001. Eligible cohorts vary with cadence.')
finish(s,'Longer history gives the clearest improvement for ≥C',90,"""
This is the strongest result in the study. With sixteen frames, C-or-stronger TSS rises from 0.674 for fifteen hours, to 0.742 for thirty hours, and 0.789 for forty-five hours. The endpoint difference is an absolute TSS gain of 0.115. All three pairwise TSS comparisons are significant in the paper’s paired tests. ROC-AUC also rises, from 0.910 to 0.958, so the improvement is not visible only at one operating threshold. For M-or-stronger, the means move in the same direction, but uncertainty is much larger and the TSS differences are not significant. We therefore emphasize the C-task evidence. The result is consistent with useful information in longer active-region evolution. It does not identify an optimal cadence in general: we are comparing three settings, and requiring a longer complete history changes the eligible sequence population. A follow-up using matched prediction endpoints would isolate the temporal effect more tightly.
""",'Give the audience time to read the teal progression. Say “0.115 absolute TSS,” not “11.5 percent more accurate.”',PAPER+' §5.3, Table 4. Repository aggregation of fixed_TSS reproduced all six means and sample SDs to 3 decimals.', 'Cadence and span change together in this experiment. The cohort-count backup shows sample attrition. These results support, but do not prove a purely causal temporal-span explanation.')

# 10 — Frames
s=start('Four observations can preserve a useful 45-hour history','04 / FRAME COUNT')
for j,(i,t) in enumerate([(0,'−45 h'),(5,'−30 h'),(10,'−15 h'),(15,'Now')]):
    pic(s,ASSETS/f'frame_{i:02d}.png',.76+j*1.31,2.44,1.15,1.15)
    txt(s,t,.73+j*1.31,3.8,1.22,.32,15,INK,True,align='center')
txt(s,'75% fewer input frames',.79,4.57,5.25,.72,25,TEAL,True)
txt(s,'4 instead of 16\nSame 45-hour endpoints',.79,5.44,5.14,.8,20,INK)
legend(s,6.54,1.99)
chart(s,6.16,2.7,6.35,3.58,['16','8','4'],[('C',[.789,.802,.805],[.021,.033,.025],TEAL),('M',[.706,.751,.752],[.107,.075,.069],ORANGE)],ymin=.55,ymax=.9,ticks=[.6,.7,.8,.9],value_labels=True,xlabel='Frames over the same 45-hour window')
source(s,'Source: paper Table 5. Mean ±1 SD over 10 seeds. TSS differences were not significant. Real example: AR 1339, 4–6 Nov 2011 (TAI).')
finish(s,'Four observations can preserve a useful 45-hour history',85,"""
Now we keep the forty-five-hour span and reduce the frame count. The images on the left are actual observations from the repository, illustrating the four timestamps. They are an input example, not an explanation of a particular model prediction. On the right, C-or-stronger TSS is 0.789 with sixteen frames, 0.802 with eight, and 0.805 with four. For M-or-stronger, the means are 0.706, 0.751 and 0.752. The TSS differences are not significant, so I would describe the sparse settings as competitive, rather than claim four frames are better. The practical result is that four frames can preserve useful information across the long window while reducing the number of input frames by seventy-five percent. That percentage describes the input count. We did not benchmark wall-clock latency, memory or energy, and their savings need not scale linearly with frame count.
""",'Keep attention on preserved temporal endpoints and competitive scores. Do not imply formal statistical equivalence.',PAPER+' §5.4, Table 5. AR1339 frames copied unchanged from data/wds_out/train/shard-000015.tar through shard-000017.tar.', '75% = (16−4)/16. Lack of statistical significance is not an equivalence or non-inferiority test. No measured runtime or memory savings are claimed.')

# 11 — Metric tradeoff
s=start('A higher mean TSS does not settle the operational choice','INTERPRETING THE FRAME-COUNT RESULT')
txt(s,'≥M task  •  fixed 45-hour window',.74,1.96,11.8,.44,21,INK,True)
for j,(metric,vals,ylim,ticks,col) in enumerate([
    ('TSS',[.706,.751,.752],(.65,.8),[.65,.7,.75,.8],TEAL),
    ('PR-AUC',[.393,.389,.359],(.32,.42),[.32,.36,.4],ORANGE),
    ('F1',[.362,.352,.327],(.30,.39),[.3,.34,.38],ORANGE)]):
    x=.71+j*4.23
    rect(s,x,2.73,3.99,2.94,WHITE)
    chart(s,x+.04,3.03,3.9,2.25,['16','8','4'],[(metric,vals,[0]*3,col)],ymin=ylim[0],ymax=ylim[1],ticks=ticks,value_labels=True,ylabel=metric)
band(s,'Sparse inputs are viable; the preferred frame count depends on the metric and task.',6.05)
source(s,'Source: paper Table 5. Descriptive means only; axes differ between panels. Full standard deviations are in backup B05.')
finish(s,'A higher mean TSS does not settle the operational choice',55,"""
The M-or-stronger task illustrates why we should look beyond a single score. Moving from sixteen to four frames raises mean TSS, but mean PR-AUC falls from 0.393 to 0.359 and mean F1 falls from 0.362 to 0.327. These panels show descriptive means on different vertical scales; they are not additional significance tests. TSS measures a balance of hit and false-positive rates at the chosen threshold. PR-AUC summarizes precision-recall ranking, and F1 emphasizes precision and recall at the operating point. An application concerned with missed strong flares or the reliability of alerts may therefore choose differently. Our conclusion is that sparse inputs are viable within this benchmark, with metric-dependent tradeoffs. We have not established a universal best frame count or an operational alert policy.
""",'Explicitly say the axes differ. Leave uncertainty and full tables for backup B05.',PAPER+' §5.4, Table 5.', 'TSS is not alert precision. Calibration, deployment prevalence, costs of missed events and false alarms, and a pre-specified operating policy would matter for operational deployment.')

# 12 — Scope
s=start('What the evidence supports—and what to test next','SCOPE & NEXT EXPERIMENTS')
for y,n,obs,nxt in [
    (2.12,'01','Temporal coverage is promising','Match prediction endpoints across all cadences'),
    (3.37,'02','Sparse long-window inputs are competitive','Measure latency, memory and non-inferiority'),
    (4.62,'03','Results hold within this benchmark','Evaluate new regions, cycles and magnetogram data')]:
    txt(s,n,.75,y,.65,.55,26,TEAL,True)
    txt(s,obs,1.67,y+.01,10.87,.45,23,INK,True)
    txt(s,nxt,1.68,y+.59,10.75,.5,19,MUTED)
band(s,'Single-image vs. sequence scores are suggestive: their eligible sample sets are different.',6.15)
source(s,'Source: paper §6; repository sequence-count audit. Seed variation does not measure generalization uncertainty across solar cycles.')
finish(s,'What the evidence supports—and what to test next',70,"""
There are three natural next steps. First, evaluate different temporal windows on a common set of prediction endpoints. Our current sequence selection requires complete history, and the number and composition of eligible examples change with cadence. A matched cohort would separate that selection effect from the value of a longer history. Second, benchmark the practical cost of the sparse configurations and test a pre-specified non-inferiority margin if the goal is to claim comparable performance. Third, validate outside this benchmark, including other solar-cycle intervals and independent magnetogram data. There are also domain issues such as overlapping active-region content and the use of natural-image or video pretraining. Finally, the image and sequence experiments do not form a fully matched architecture comparison. A clean baseline would train and evaluate a single-image model on the last frames of the exact same valid sequences.
""",'Deliver these as the next experimental design, not as an apology. Backup B06 quantifies the changing cohorts.',PAPER+' §6; results/a3_sequence_size_counts.tsv.', 'A stronger static baseline uses the final frame of the same sequence-derived examples with matched labels, splits and training budget. A solar-cycle holdout probes a different generalization question from the current AR-disjoint split.')

# 13 — Conclusion and questions
s=start('Choose the history before increasing the sampling density','TAKEAWAYS',dark=True)
for y,num,head,body in [
    (2.23,'01','Longer history gave the clearest gain','≥C TSS: 0.674 → 0.789 across the tested 16-frame settings.'),
    (3.58,'02','Sparse sampling remained competitive','Four frames over 45 hours retained useful predictive performance.'),
    (4.93,'03','Input design deserves explicit evaluation','Spatial detail, disk position and metric choice affect the conclusion.')]:
    txt(s,num,.72,y,.75,.5,27,'71D1C7',True)
    txt(s,head,1.73,y,10.8,.6,24,WHITE,True)
    txt(s,body,1.74,y+.64,10.72,.49,17,'BBD0DD')
txt(s,'Thank you  •  Questions',.74,6.49,7,.34,19,'71D1C7',True)
txt(s,'danil.garmaev@umontreal.ca',8.45,6.54,4.22,.32,14,'BBD0DD',align='right')
finish(s,'Choose the history before increasing the sampling density',45,"""
I’ll leave you with three takeaways. First, extending the observed history gave our clearest improvement, especially for C-or-stronger prediction. Second, once a forty-five-hour window was available, four or eight frames remained competitive with sixteen. Third, input design needs to be evaluated directly: more pixels, more frames or more specialized training do not automatically produce a better predictor across tasks and metrics. The practical message is to establish the useful temporal coverage, then test how sparsely we can observe it. These are benchmark findings with a clear next step: matched cohorts and measured computational costs. Thank you. I’m happy to take questions.
""",'Finish here at approximately 14 minutes. Leave this slide visible during questions. Backup slides are hidden from normal slideshow navigation.',PAPER+' §7; Tables 2–5.')

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
s=start('Longer complete histories reduce the eligible sample pool','BACKUP B06 / SUPPLEMENTARY REPOSITORY ANALYSIS',backup=True)
table(s,['16-frame cadence','History','Train eligible','Test eligible','≥M test positives'],[
    ['60 min','15 h','124,101','15,602','531'],
    ['120 min','30 h','50,094','6,349','201'],
    ['180 min','45 h','26,825','3,414','105'],
],y=2.2,row_h=.6,col_widths=[2.68,1.38,2.66,2.66,2.62],size=16)
txt(s,'Why this matters',.78,5.0,4.4,.5,24,TEAL,True)
txt(s,'Cadence comparisons also change the eligible cohort and rare-event counts.\nNext control: use shared prediction endpoints across every temporal setting.',.78,5.72,11.73,.77,19,INK)
source(s,'Source: results/a3_sequence_size_counts.tsv (cached repository snapshot). Train counts are before negative subsampling; not new model results.')
finish(s,'B06 — Eligible sequence cohorts',0,'This is supplementary analysis from the repository, not a table in the accepted paper. It documents how requiring longer histories changes sample eligibility. The cached audit reports 15,602 test sequences at sixty minutes versus 3,414 at a hundred and eighty minutes; M-or-stronger positives fall from 531 to 105. This is a reason to avoid attributing every observed gain solely to temporal span. These are sequence examples and can be correlated within active regions, not counts of independent flare events. The file is a cached snapshot; we did not regenerate it or train new models for the presentation.',source='results/a3_sequence_size_counts.tsv; scripts/count_a3_sequence_sizes.py. Counts read directly from cached TSV; provenance recorded in analysis notes.',qa='Do not use these as exact counts of independent events. The 60-minute effective training count after negative subsampling is 120,203 in this snapshot, compared with 124,101 before subsampling.',backup=True)

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
txt(s,'Visuals: paper Fig. 1; real AR 1339 images from repository WebDataset shards.\nAll plots redrawn from accepted-paper values; supplementary cohort counts identified separately.',.76,6.36,11.8,.44,11,MUTED)
finish(s,'B08 — References and visual provenance',0,'These references are copied from the accepted paper. The main charts are newly drawn for readability using the paper’s numeric results. The magnetograms are actual data, not generated scientific images. The cohort analysis is explicitly identified as supplementary repository material.',source=PAPER+' references 1–3 and 17; presentation/assets/frame_provenance.json.',backup=True)

# Hide appendix from a normal oral-presentation slideshow, but retain navigation.
for s in slides[13:]:s._element.set('show','0')
prs.save(OUT/'ICANN_2026_Solar_Flare_Prediction.pptx')
pdf.save()
(OUT/'speaker_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
(OUT/'layout_checks.json').write_text(json.dumps(text_checks,ensure_ascii=False,indent=2))
md=['# ICANN 2026 speaker guide','', 'Paper 750 · Danil Garmaev and Jean Meunier','',
    '13 main slides · target 14 minutes · 1 minute buffer within the 15-minute slot · 5 minutes for questions.',
    '', 'The same notes are embedded in the PowerPoint Notes pane. Backup slides are hidden from normal slideshow progression; access them through PowerPoint’s slide navigator.', '']
elapsed=0
for n in notes:
    if not n['backup']:
        end=elapsed+n['seconds'];timing=f"{elapsed//60}:{elapsed%60:02d}–{end//60}:{end%60:02d}";elapsed=end
    else:timing='Q&A backup'
    md += [f"## {n['slide']:02d}. {n['title']} ({timing})",'',n['script'],'']
    for k,title in [('cue','Delivery'),('qa','If asked'),('source','Source')]:
        if n[k]:md += [f'**{title}:** {n[k]}','']
(OUT/'Speaker_Notes_and_Rehearsal.md').write_text('\n'.join(md))
overflow=[r for r in text_checks if r['need_pt']>r['height_pt']+1]
print(json.dumps({'slides':len(slides),'main_slides':13,'timing_seconds':elapsed,'overflow_count':len(overflow),'overflows':overflow},ensure_ascii=False,indent=2))
