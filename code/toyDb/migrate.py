import logging

from databases.ShaderDb import ShaderDB, SHADER_DIR

def construct_path(shaderID: str):
    mangled_name = ShaderDB.convert_to_mangled_format(shaderID)
    path = f"{SHADER_DIR}/../shaders-new/json/{mangled_name[0]}-{mangled_name[7]}/{mangled_name[1]}-{mangled_name[8]}/{mangled_name}.json"
    return path

def construct_path(shaderID: str):
    mangled_name = ShaderDB.convert_to_mangled_format(shaderID)
    path = f"{SHADER_DIR}/../shaders-new/json/{mangled_name[0]}-{mangled_name[7]}/{mangled_name[1]}-{mangled_name[8]}/{mangled_name}.json"

    return path

def migrate():
    shaderDB = ShaderDB(SHADER_DIR)
    shaderDB.scan_local(mangledName=False)
    shaderDB.load_all()

    shaderDB.write_shaders(construct_path)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG if False else logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    migrate()