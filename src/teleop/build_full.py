import xml.etree.ElementTree as E,glob,os
f=glob.glob('generated_urdf/body_source/**/*.urdf',recursive=True)[0]; src=E.parse(f).getroot(); root=E.Element('robot',{'name':'tianji_humanoid_tapcap'});
for x in src: root.append(x)
fg=glob.glob('generated_urdf/**/从夹爪组件0720/*.urdf',recursive=True)[0]; gr=E.parse(fg).getroot()
for side,parent in [('left','TCP_Link_L'),('right','TCP_Link_R')]:
 for x in gr.findall('link'):
  x=E.fromstring(E.tostring(x));x.attrib['name']=side+'_follower_'+x.attrib['name']
  for m in x.findall('.//mesh'):
   m.attrib['filename']=os.path.abspath(glob.glob('generated_urdf/**/'+os.path.basename(m.attrib['filename']),recursive=True)[0])
  root.append(x)
 for x in gr.findall('joint'):
  x=E.fromstring(E.tostring(x));x.attrib['name']=side+'_follower_'+x.attrib['name'];x.find('parent').attrib['link']=side+'_follower_'+x.find('parent').attrib['link'];x.find('child').attrib['link']=side+'_follower_'+x.find('child').attrib['link'];root.append(x)
 j=E.SubElement(root,'joint',{'name':side+'_follower_mount','type':'fixed'});E.SubElement(j,'parent',{'link':parent});E.SubElement(j,'child',{'link':side+'_follower_base_link'});E.SubElement(j,'origin',{'xyz':'0 0 0','rpy':'0 0 0'})
for m in root.findall('.//mesh'):
 if not os.path.isabs(m.attrib['filename']):
  h=glob.glob('generated_urdf/**/'+os.path.basename(m.attrib['filename']),recursive=True)
  if h:m.attrib['filename']=os.path.abspath(h[0])
E.ElementTree(root).write('generated_urdf/tianji_humanoid_tapcap.urdf',encoding='utf-8',xml_declaration=True)
