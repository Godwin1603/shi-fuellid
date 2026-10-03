import urllib.request, re
req = urllib.request.Request('https://www.nvidia.com/Download/processFind.aspx?psid=74&pfid=750&osid=57&lid=1&whql=1&lang=en-us&ctk=0&qnf=0', headers={'User-Agent': 'Mozilla/5.0'})
data = urllib.request.urlopen(req).read().decode('utf-8')
matches = re.findall(r'driverResults\.aspx/(\d+)/en-us', data)
if matches:
    first_id = matches[0]
    print('Found driver ID:', first_id)
    req2 = urllib.request.Request('https://www.nvidia.com/download/driverResults.aspx/' + first_id + '/en-us', headers={'User-Agent': 'Mozilla/5.0'})
    data2 = urllib.request.urlopen(req2).read().decode('utf-8')
    dl = re.search(r'//[^ "'<>]+?\.exe', data2)
    if dl:
        print('https:' + dl.group(0))
