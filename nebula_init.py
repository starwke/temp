import os

from configs.user_config import GRAPHD_HOST, NEBULA_PASSWORD
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

# Nebula Graph 数据库连接配置
config = Config()
config.max_connection_pool_size = 10
# 使用环境变量中的地址和密码
host = GRAPHD_HOST
port = 9669  # 或者任何 Nebula Graph 正在监听的端口
user = "root"
password = NEBULA_PASSWORD

# 创建连接池
connection_pool = ConnectionPool()
if not connection_pool.init([(host, port)], config):
    print("Failed to connect to Nebula Graph")
    exit(1)

# 获取一个连接
session = connection_pool.get_session(user, password)
if session is None:
    print("Failed to get session")
    exit(1)

# 选择图空间
space_name = "phillies_rag"  # 替换为您的图空间名称
use_space_query = f"USE {space_name};"
session.execute(use_space_query)
# 执行查询

# query = "SHOW SPACES"
query = """
MATCH (p:`entity`)-[e:relationship]->(m:`entity`)
WHERE p.`entity`.`name` == 'Philadelphia Phillies'
RETURN p, e, m;
"""
result = session.execute(query)

# 检查查询是否成功
if result.is_succeeded():
    print(f"Query succeeded: {result}")
else:
    print(f"Query failed: {result.error_msg()}")

# 关闭会话和连接池
session.release()
connection_pool.close()
