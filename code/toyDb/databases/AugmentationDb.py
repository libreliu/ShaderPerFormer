import peewee as pw
import os

# Define the database
dbA = pw.SqliteDatabase(None)

def init_from_default_dbA():
    dbA.init(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../augmentation.db"), pragmas={'foreign_keys': 1})

class BaseModel(pw.Model):
    class Meta:
        database = dbA


class ImageOnlyShader(BaseModel):
    shader_id = pw.CharField()
    # Store as spirv blob for more accurate representation
    fragment_spv = pw.BlobField()


# Define the model for the "augmentation" table
class Augmentation(BaseModel):
    # Define the columns of the table
    augmentation = pw.IntegerField()
    augmentation_annotation = pw.CharField(null=True)
    fragment_spv = pw.BlobField()
    shader_id = pw.CharField()
    depth = pw.IntegerField()
    parent = pw.ForeignKeyField('self', backref='children', null=True)
    dis_from_last = pw.IntegerField()
    dis_ratio_from_last = pw.FloatField()
    dis_from_origin = pw.IntegerField()
    dis_ratio_from_origin = pw.FloatField()


# Create the table if it doesn't exist

