import os
import logging
from typing import List
import requests, json
import functools

logger = logging.getLogger(__name__)

SHADER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../dataset/shaders"
)

def cachable_attribute(func):
    """Will handle load & cache automatically"""
    
    @functools.wraps(func)
    def wrapped_method(self: 'ShaderProxy'):
        bypass_cache = self.db is None
        cached_result = self.db.get_cache(self.id, func.__name__) if not bypass_cache else None
        if cached_result is None:
            # leave the load status as it is
            unload_later = False

            if self.json is None:
                self.load()
                unload_later = True

            result = func(self)
            if not bypass_cache:
                self.db.put_cache(self.id, func.__name__, result)
            
            if unload_later:
                self.unload()

            return result
        else:
            return cached_result

    return wrapped_method


class ShaderProxy:
    def __init__(self, db: 'ShaderDB', id, path=None, remote=False, expandPath=True):
        self.db = db
        self.id = id
        self.remote = remote
        self.json = None

        if (not remote) and expandPath:
            assert(path is not None)
            self.path = os.path.realpath(path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShaderProxy):
            return False

        return self.id == other.id and self.path == other.path
    
    def get_slim_version(self) -> 'ShaderProxy':
        """
        A slim version, with no reference to db object involved;
        This greatly reduce the overhead of pickling, but not all functions may be possible
        """
        assert(not self.remote)
        slim = ShaderProxy(None, self.id, self.path, self.remote, expandPath=False)
        slim.json = self.json

        return slim

    def load(self):
        assert(not self.remote)
        if self.json is not None:
            return

        with open(self.path, 'r') as f:
            self.json = json.load(f)

    def unload(self):
        self.json = None

    def make_offline(self, apiKey, requestSession=None):
        """Use requestSession to reuse connection"""
        if not self.remote:
            return
        
        sess = requestSession if requestSession is not None else requests.Session()
        r = sess.get(
            f"https://www.shadertoy.com/api/v1/shaders/{self.id}?key={apiKey}",
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            }
        )

        res = r.json()
        offlinePath = os.path.realpath(self.db.make_shader_path(self.id))
        self.db.create_path_dirs_if_nonexist(offlinePath)
        with open(offlinePath, 'w') as f:
            json.dump(res, f)

        self.remote = False
        self.path = offlinePath
        logger.info(f"Downloaded shader {offlinePath}")

    def write_to(self, path):
        assert(not self.remote)

        with open(path, 'w') as f:
            json.dump(self.json, f)
        
        logger.info(f"Write {self.id} to {path}")

    def prepare_assets(self, requestSession=None):
        self.load()

        for renderpass in self.json["Shader"]["renderpass"]:
            for inputChannel in renderpass["inputs"]:
                src = inputChannel["src"]

                possibleExtensions = {
                    "texture": (".jpg", ".png"),
                    "music": (".mp3",),
                    "cubemap": (".jpg", ".png")
                }

                if not inputChannel["ctype"] in possibleExtensions:
                    logger.warn(f"{self.id} input channel unknown ctype: {inputChannel['ctype']}, skip")
                    continue
                else:
                    valids = possibleExtensions[inputChannel["ctype"]]
                    valid = False
                    for suffix in valids:
                        if src.endswith(suffix):
                            valid = True
                            break
                    
                    if not valid:
                        raise Exception(f"{self.id} input channel (ctype={inputChannel['ctype']}) src not ended in {valids}: {src}")
                
                self.try_download_media(src, requestSession=requestSession)
        
        self.unload()

    def try_download_media(self, mediaURL, replace=False, chunkSize=8192, requestSession=None):
        if not mediaURL.startswith("/media/a/"):
            raise Exception("media URL not valid")

        fileName = mediaURL[9:]
        stat = None
        try:
            stat = os.stat(f"{self.shaderDir}/media/a/{fileName}")
        except FileNotFoundError:
            pass

        if stat is not None and not replace:
            logger.info(f"{fileName} is already there, skip")
            return

        logger.info(f"Downloading {mediaURL}...")
        sess = requestSession if requestSession is not None else requests.Session()
        r = sess.get(
            f"https://www.shadertoy.com{mediaURL}",
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            }
        )
        
        mediaPath = os.path.realpath(os.path.join(self.db.shaderDir, f"./media/a/{fileName}"))
        try:
            os.makedirs(os.path.dirname(mediaPath))
        except FileExistsError:
            pass
        with open(mediaPath, 'wb') as fd:
            for chunk in r.iter_content(8192):
                fd.write(chunk)

    def is_valid(self):
        assert(self.json is not None)

        # {"Error": "Shader not found"}
        if "Error" in self.json.keys():
            return False
        else:
            return True

    @cachable_attribute
    def is_imageonly(self):
        """Check if the image have got only one pass"""
        imageOnly = True
        for rpass in self.json["Shader"]["renderpass"]:
            if rpass["type"] not in ("image", "common"):
                imageOnly = False
                break

        return imageOnly
    
    # TODO: check if only 1 image pass
    def get_renderpass_index(self, passType):
        assert(self.json is not None)
        for passIdx, pass_ in enumerate(self.json["Shader"]["renderpass"]):
            if pass_["type"] == passType:
                return passIdx

        return None

    def get_renderpass(self, passType):
        assert(self.json is not None)
        passIdx = self.get_renderpass_index(passType)
        return self.json["Shader"]["renderpass"][passIdx] if passIdx is not None else None

class ShaderDB:
    def __init__(self, shaderDir, attrCacheName='attributes.json', flushAttrCache=False):
        """NOTE: attrCacheName shall not be conflict with shader json names"""
        self.shaderDir = os.path.realpath(shaderDir)
        self.offlineShaders = {}

        self.attrCacheName = os.path.basename(attrCacheName)
        self.attrCachePath = os.path.join(self.shaderDir, attrCacheName)
        self.attrCache = {}

        if not flushAttrCache:
            try:
                with open(self.attrCachePath, 'r') as f:
                    self.attrCache = json.load(f)
            except FileNotFoundError:
                logger.info(f"Shader attribute cache {self.attrCachePath} not found, continue.")
        else:
            logger.info(f"Skipped loading caches as instructed. This will flush the cache when program ends.")

    def get(self, shaderID):
        return self.offlineShaders[shaderID]

    def save_cache(self):
        with open(self.attrCachePath, 'w') as f:
            json.dump(self.attrCache, f)

    def get_cache(self, shaderID, attrName):
        if shaderID not in self.attrCache:
            return None
        
        if attrName not in self.attrCache[shaderID]:
            return None

        return self.attrCache[shaderID][attrName]

    def put_cache(self, shaderID, attrName, attrVal):
        if shaderID not in self.attrCache:
            self.attrCache[shaderID] = {}

        self.attrCache[shaderID][attrName] = attrVal

    def make_shader_path(self, shaderID):
        """Generate a path relative to shaderDir for storing the shader"""
        path = f"{self.shaderDir}/json/" + \
               f"{shaderID[0]}/{shaderID[1]}/{shaderID[2:]}/{shaderID}.json"
        
        path = os.path.realpath(path)
        return path

    def scan_local_legacy(self):
        shaderDirFiles = os.listdir(f'{self.shaderDir}/jsonLegacy')
        for fileName in shaderDirFiles:
            if fileName.endswith(".json") and len(fileName) >= 11:
                shaderID = fileName[:fileName.index('-')]
                assert(shaderID not in self.offlineShaders)
                self.offlineShaders[shaderID] = ShaderProxy(
                    self,
                    shaderID,
                    fileName
                )
        
        logger.info(f"Loaded {len(self.offlineShaders)} offline shaders.")
    
    def load_all(self):
        for shaderID, shaderProxy in self.offlineShaders.items():
            shaderProxy.load()
    
    def unload_all(self):
        for shaderID, ShaderProxy in self.offlineShaders.items():
            ShaderProxy.unload()

    @staticmethod
    def convert_from_mangled_format(mangledName: str):
        if len(mangledName) != 13 or '-' not in mangledName or mangledName[6] != '-':
            raise Exception(
                "Mangled format expects things like 3dXttZ-NLULLU (N: Numeric, L: Lower case, U: Upper case)"
            )

        unfixedId = mangledName[:6]
        fixedId = ""
        for idx, fixMethod in enumerate(mangledName[7:]):
            if fixMethod == 'N':
                if not ('0' <= unfixedId[idx] <= '9'):
                    raise Exception(f"Expect numeric on pos {idx}")
                else:
                    fixedId += unfixedId[idx]
            elif fixMethod == 'L' or fixMethod == 'U':
                if not ('A' <= unfixedId[idx] <= 'Z' or 'a' <= unfixedId[idx] <= 'z'):
                    raise Exception(f"Unknown character appeared on pos {idx}: {unfixedId[idx]}")
                
                if fixMethod == 'L':
                    fixedId += unfixedId[idx].lower()
                else:
                    fixedId += unfixedId[idx].upper()
        
        assert(len(fixedId) == 6)
        return fixedId

    @staticmethod
    def convert_to_mangled_format(unmangledName: str):
        assert(len(unmangledName) == 6)
        suffix = ""
        for idx, ch in enumerate(unmangledName):
            if 'A' <= ch <= 'Z':
                suffix += 'U'
            elif 'a' <= ch <= 'z':
                suffix += 'L'
            elif '0' <= ch <= '9':
                suffix += 'N'
            else:
                raise Exception(f"Unexpected character {ch} on pos {idx}")

        assert(len(suffix) == 6)
        return f"{unmangledName}-{suffix}"

    def scan_local(self, mangledName=True):
        """Name mangling: 3dXttZ => 3N/dU/XU/3dXttZ-NLULLU.json"""
        for root, dirs, files in os.walk(self.shaderDir):
            for fileName in files:
                if fileName.endswith('.json') and fileName != self.attrCacheName:
                    if mangledName:
                        shaderID = ShaderDB.convert_from_mangled_format(fileName[:fileName.index('.')])
                    else:
                        shaderID = fileName[:fileName.index('.')]

                    assert(len(shaderID) == 6 and shaderID not in self.offlineShaders)
                    self.offlineShaders[shaderID] = ShaderProxy(
                        self,
                        shaderID,
                        os.path.realpath(os.path.join(root, fileName))
                    )

        logger.info(f"Loaded {len(self.offlineShaders)} offline shaders.")

    @staticmethod
    def create_path_dirs_if_nonexist(path):
        dirpath = os.path.dirname(path)
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

    def write_shaders(self, shaderPathFn=None):
        """Write current offline shaders to somewhere else

        args:
        shaderPathFn: Function[ShaderProxy] -> str
        """
        
        for shaderID, shaderProxy in self.offlineShaders.items():
            targetPath = shaderPathFn(shaderID)
            self.create_path_dirs_if_nonexist(targetPath)
            shaderProxy.write_to(targetPath)

    def delete_shader(self, shaderID):
        assert(shaderID in self.offlineShaders)
        shaderProxy = self.offlineShaders[shaderID]
        os.remove(shaderProxy.path)
        del self.offlineShaders[shaderID]

    def update(self, apiKey):
        """Use requestSession to reuse the connection"""
        sess = requests.Session()

        r = sess.get(
            f'https://www.shadertoy.com/api/v1/shaders/?key={apiKey}',
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
            }
        )
        res = r.json()
        shaderIDs = res['Results']
        logger.info(f"Shadertoy API: Got {len(shaderIDs)} shaders.")

        onlineIDs = set(shaderIDs)
        offlineIDs = set(self.offlineShaders.keys())
        amendIDs = onlineIDs - offlineIDs

        logger.info(f"Online: {len(onlineIDs)}, Offline: {len(offlineIDs)}, Amend: {len(amendIDs)}")
    
        for shaderID in amendIDs:
            shaderProxy = ShaderProxy(self, shaderID, None, True)
            shaderProxy.make_offline(apiKey, sess)
        
            self.offlineShaders[shaderID] = shaderProxy
        
    def update_assets(self):
        sess = requests.Session()

        for shaderID, shaderProxy in self.offlineShaders.items():
            shaderProxy.prepare_assets(requestSession=sess)

    def filter_attribute(self, good_attrs: List[str]):
        results = set()
        for shaderID, shaderProxy in self.offlineShaders.items():
            good_shader = True
            for good_attr in good_attrs:
                attr_method = getattr(shaderProxy, good_attr)

                try:
                    attr_result = attr_method()
                except:
                    logger.error(f"Error while processing shader {shaderID}")
                    raise Exception(f"Error retrieving attribute of {shaderID}")

                if attr_result != True:
                    good_shader = False
                    break
            
            if not good_shader:
                continue
            
            results.add(shaderID)
        
        return results
