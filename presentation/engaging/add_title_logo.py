"""Patch the user-edited deck directly; never rebuild it from the earlier source."""
from pathlib import Path
import base64, hashlib, json, shutil, zipfile
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
deck = ROOT / 'ICANN_2026_Solar_Flare_Prediction_Engaging.pptx'
backup = ROOT / 'archive_user_edits' / '20260915_before_logo.pptx'
backup.parent.mkdir(exist_ok=True)
if backup.exists():
    raise SystemExit('Backup already exists; inspect current deck before patching again.')
shutil.copy2(deck, backup)
ns = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
rel_ns = 'http://schemas.openxmlformats.org/package/2006/relationships'
with zipfile.ZipFile(deck) as z:
    parts = {n:z.read(n) for n in z.namelist()}
    infos = z.infolist()
slide_name = 'ppt/slides/slide1.xml'
rels_name = 'ppt/slides/_rels/slide1.xml.rels'
slide = E.fromstring(parts[slide_name]); rels = E.fromstring(parts[rels_name])
shape_id = max(int(n.get('id')) for n in slide.findall('.//p:cNvPr',ns)) + 1
rid = 'rId' + str(max(int(n.get('Id')[3:]) for n in rels)+1)
image_name = 'ppt/media/udem_logo_white_official.png'
assert image_name not in parts
x,y,w = [round(v*914400) for v in (10.7,.33,2.05)]
h = round(w*134.282/351.479)
pic = E.fromstring(f'''<p:pic xmlns:p="{ns['p']}" xmlns:a="{ns['a']}" xmlns:r="{ns['r']}">
<p:nvPicPr><p:cNvPr id="{shape_id}" name="Université de Montréal — white official logo" descr="Université de Montréal institutional logo, white version from the university website."/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'''.encode())
slide.find('p:cSld/p:spTree',ns).append(pic)
E.SubElement(rels,'{'+rel_ns+'}Relationship',Id=rid,Type=ns['r']+'/image',Target='../media/udem_logo_white_official.png')
updates={slide_name:E.tostring(slide,xml_declaration=True,encoding='UTF-8',standalone=True),
         rels_name:E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)}
tmp=deck.with_suffix('.tmp.pptx')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for info in infos: z.writestr(info,updates.get(info.filename,parts[info.filename]))
    z.writestr(image_name,(ROOT/'assets/udem_logo_white_official.png').read_bytes())
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
    changed=[n for n in parts if parts[n]!=z.read(n)]
    assert set(changed)==set(updates),changed
    for n in z.namelist():
        if n.endswith(('.xml','.rels')): E.fromstring(z.read(n))
    check=E.fromstring(z.read(slide_name))
    original=E.fromstring(parts[slide_name])
    check.find('p:cSld/p:spTree',ns).remove(check.findall('.//p:pic',ns)[-1])
    assert E.tostring(check,method='c14n')==E.tostring(original,method='c14n')
tmp.replace(deck)
report={'original_backup':str(backup),'changed_existing_parts':changed,
        'all_other_slides_and_notes_byte_identical':True,'original_title_shapes_preserved':True,
        'embedded_logo':image_name,'logo_bounds_inches':[10.7,.33,2.05,h/914400],
        'logo_source':'https://www.umontreal.ca/typo3temp/udemlessc/lessphp_9477013ded911f281ce354ea476ce0ead34a363a.css?1769692499',
        'source_selector':'.udem-logo-udem-inv','brand_guidance':'https://marque.umontreal.ca/',
        'source_svg_sha256':hashlib.sha256((ROOT/'assets/udem_logo_white_official.svg').read_bytes()).hexdigest(),
        'method':'Original white SVG rasterized at 1406px width; unchanged proportions and colors.',
        'zip_and_xml_valid':True}
(ROOT/'logo_update_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))

# Companion rendering of the current title slide for inspection, not a PowerPoint export.
SVG='http://www.w3.org/2000/svg'
svg=E.Element('{'+SVG+'}svg',nsmap={None:SVG},width='1600',height='900',viewBox='0 0 960 540')
def node(tag,**attrs): return E.SubElement(svg,'{'+SVG+'}'+tag,{k.replace('_','-'):str(v) for k,v in attrs.items()})
node('rect',x=0,y=0,width=960,height=540,fill='#142B3C')
with zipfile.ZipFile(deck) as z:
    for shape in slide.find('p:cSld/p:spTree',ns):
        xf=shape.find('.//a:xfrm',ns)
        if xf is None: continue
        off=xf.find('a:off',ns); ext=xf.find('a:ext',ns)
        xx,yy,ww,hh=[int(n.get(k))/12700 for n,k in [(off,'x'),(off,'y'),(ext,'cx'),(ext,'cy')]]
        kind=E.QName(shape).localname
        if kind=='pic':
            r=shape.find('.//a:blip',ns).get('{'+ns['r']+'}embed')
            target=next(a.get('Target') for a in rels if a.get('Id')==r)
            b=z.read('ppt/'+target.removeprefix('../'))
            node('image',x=xx,y=yy,width=ww,height=hh,href='data:image/png;base64,'+base64.b64encode(b).decode())
        elif kind=='cxnSp':
            c=shape.find('.//a:ln/a:solidFill/a:srgbClr',ns)
            line=shape.find('.//a:ln',ns)
            node('line',x1=xx,y1=yy,x2=xx+ww,y2=yy+hh,stroke='#'+(c.get('val') if c is not None else '385061'),stroke_width=int(line.get('w','12700'))/12700)
        elif kind=='sp':
            pos_y=yy
            for para in shape.findall('p:txBody/a:p',ns):
                default=para.find('a:pPr/a:defRPr',ns)
                sz=int(default.get('sz','1800'))/100 if default is not None else 18
                c=default.find('a:solidFill/a:srgbClr',ns) if default is not None else None
                color='#'+(c.get('val') if c is not None else 'FFFFFF')
                bold=default is not None and default.get('b')=='1'
                pp=para.find('a:pPr',ns); align=pp.get('algn','l') if pp is not None else 'l'
                text=node('text',x=xx+(ww if align=='r' else 0),y=pos_y+sz*.83,font_family='Liberation Sans, Arial, sans-serif',font_size=sz,font_weight='bold' if bold else 'normal',fill=color,text_anchor='end' if align=='r' else 'start')
                text.text=''.join(para.xpath('.//a:t/text()',namespaces=ns))
                spacing=para.find('a:pPr/a:lnSpc/a:spcPts',ns)
                pos_y+=int(spacing.get('val'))/100 if spacing is not None else sz*1.16
(ROOT/'Title_Slide_Logo_Preview.svg').write_bytes(E.tostring(svg,xml_declaration=True,encoding='UTF-8'))
