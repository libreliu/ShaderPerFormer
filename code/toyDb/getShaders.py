import requests, json, os
import argparse

# https://www.shadertoy.com/howto

def sanitize_name(name: str):
    out = ""
    for ch in name:
        if ch.isalpha() or ch.isdigit():
            out += ch
        elif ch == ' ':
            out += '-'
        else:
            pass
    return out

def get_shader_online(shaderID, apiKey):
    r = requests.get(
        f"https://www.shadertoy.com/api/v1/shaders/{shaderID}?key={apiKey}",
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }
    )
    res = r.json()

    print(f'Shader {shaderID}: name={res["Shader"]["info"]["name"]}, author={res["Shader"]["info"]["username"]}')

    with open(f'shaders/json/{shaderID}-{sanitize_name(res["Shader"]["info"]["name"])}.json', 'w') as f:
        json.dump(res, f)

def download_media(mediaURL: str, replace=False, chunkSize=8192):
    if not mediaURL.startswith("/media/a/"):
        raise Exception("media URL not valid")

    fileName = mediaURL[9:]
    stat = None
    try:
        stat = os.stat(f"shaders/media/a/{fileName}")
    except FileNotFoundError:
        pass

    if stat is not None and not replace:
        print(f"- {fileName} is already there, skip")
        return

    print(f"- Downloading {mediaURL}...")
    r = requests.get(
        f"https://www.shadertoy.com{mediaURL}",
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }
    )

    with open(f"shaders/media/a/{fileName}", 'wb') as fd:
        for chunk in r.iter_content(8192):
            fd.write(chunk)

def enumerate_shaders(apiKey):
    r = requests.get(
        f'https://www.shadertoy.com/api/v1/shaders/?key={apiKey}',
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }
    )
    res = r.json()

    shaderIDs = res['Results']
    print(f"Shadertoy API: Got {len(shaderIDs)} shaders.")

    return shaderIDs

def enumerate_shaders_offline(fullName = False):
    shaderDirFiles = os.listdir('shaders/json')
    shaderIDs = []
    for fileName in shaderDirFiles:
        if fileName.endswith(".json") and len(fileName) >= 11:
            shaderID = fileName[:6]
            shaderIDs.append((shaderID, fileName))
    
    return shaderIDs

def read_shader(shaderFilename):
    with open(f'shaders/json/{shaderFilename}') as f:
        return json.load(f)

def amend_shader(apiKey):
    onlineShaderIDs = enumerate_shaders(apiKey)
    offlineShaderIDs = [idFilename[0] for idFilename in enumerate_shaders_offline()]

    amendShaders = set(onlineShaderIDs) - set(offlineShaderIDs)

    print(f"Online: {len(onlineShaderIDs)}, Offline: {len(offlineShaderIDs)}, Amend: {len(amendShaders)}")

    for shaderID in amendShaders:
        print(f"Processing {shaderID}...")
        try:
            get_shader_online(shaderID, apiKey)
        except:
            print(f"Failed to get shader {shaderID}, ignore")

def download_all_media():
    for shaderID, shaderFilename in enumerate_shaders_offline(True):
        print(f"Processing {shaderFilename}...")
        shader = read_shader(shaderFilename)
        for renderpass in shader["Shader"]["renderpass"]:
            for inputChannel in renderpass["inputs"]:
                src = inputChannel["src"]

                possibleExtensions = {
                    "texture": (".jpg", ".png"),
                    "music": (".mp3",),
                    "cubemap": (".jpg", ".png")
                }

                if not inputChannel["ctype"] in possibleExtensions:
                    print(f"{shaderID} input channel unknown ctype: {inputChannel['ctype']}, skip")
                    continue
                else:
                    valids = possibleExtensions[inputChannel["ctype"]]
                    valid = False
                    for suffix in valids:
                        if src.endswith(suffix):
                            valid = True
                            break
                    
                    if not valid:
                        raise Exception(f"{shaderID} input channel (ctype={inputChannel['ctype']}) src not ended in {valids}: {src}")
                
                download_media(src)

def get_renderpass_multiple(shader, type):
    res = []
    for rpass in shader["Shader"]["renderpass"]:
        if rpass["type"] == type:
            res.append(rpass)
    return res

def filter_imageonly_shaders():
    """Shaders that only have image (optionally, common) pass present"""
    res = []
    for shaderID, shaderFilename in enumerate_shaders_offline(True):
        shader = read_shader(shaderFilename)
        imageOnly = True
        for rpass in shader["Shader"]["renderpass"]:
            if rpass["type"] not in ("image", "common"):
                imageOnly = False
                break
        
        if imageOnly:
            res.append(shaderID)
    
    return res

if __name__ == "__main__":
    apiKey = None

    parser = argparse.ArgumentParser(prog = 'ShaderDB')
    parser.add_argument('--apiKeyFile', default="apikey.txt")
    parser.add_argument('--amend', action='store_true', help="Update online before querying")
    parser.add_argument('verb', choices=["noop", "imageonly-shaders", "check-channel-type"], default="noop")
    args = parser.parse_args()

    if args.amend:
        with open(args.apiKeyFile, 'r', encoding='utf8') as f:
            apiKey = f.read()

        print(f"Using API key: {apiKey}")
        amend_shader(apiKey)

        download_all_media()
    
    if args.verb == "imageonly-shaders":
        imageOnlyShaderIDs = filter_imageonly_shaders()
        print(f"Total: {len(imageOnlyShaderIDs)}")
