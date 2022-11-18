from claims_topics.services.file import YAMLservice
from claims_topics.config import global_config as glob

my_yaml = YAMLservice(path='claims_topics/config/input_output.yaml')
io = my_yaml.doRead()
