#!/bin/bash
# PostgreSQL 自动备份脚本
# 保留最近 7 天的备份，每天凌晨 3 点执行

BACKUP_DIR="/opt/fitness/volumes/backups"
CONTAINER="fitness-postgres"
DB_NAME="fitness"
DB_USER="postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${DATE}.sql.gz"

# 创建备份目录
mkdir -p ${BACKUP_DIR}

# 执行备份（在容器内 pg_dump，压缩后存到宿主机）
docker exec ${CONTAINER} pg_dump -U ${DB_USER} -d ${DB_NAME} | gzip > ${BACKUP_FILE}

# 检查备份是否成功
if [ $? -eq 0 ]; then
    echo "[$(date)] 备份成功: ${BACKUP_FILE} ($(du -h ${BACKUP_FILE} | cut -f1))"
else
    echo "[$(date)] 备份失败!" >&2
    exit 1
fi

# 删除 7 天前的备份
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete

# 显示当前备份列表
echo "[$(date)] 当前备份:"
ls -lh ${BACKUP_DIR}/*.sql.gz 2>/dev/null | tail -5
