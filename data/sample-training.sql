-- ============================================================
-- AI 健身教练 — 示例训练数据
-- 基于一个虚拟用户「小王」的 2 周训练记录
-- 用于演示和本地测试
-- 注意：不是真实用户数据，但训练安排符合合理的分化逻辑
-- ============================================================

-- 1. 创建一个测试用户
INSERT INTO users (username, password_hash, display_name, height_cm, weight_kg, age, goal, experience, equipment)
VALUES (
    'test_user',
    '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',  -- sha256('password')
    '小王',
    175,
    72,
    26,
    '增肌增力',
    '中级（1年）',
    '健身房全套'
);

-- 2. 用户档案
INSERT INTO user_profiles (user_id, height_cm, weight_kg, age, goal, experience_level, equipment)
VALUES (
    (SELECT id FROM users WHERE username = 'test_user'),
    175, 72, 26, '增肌增力', '中级（1年）', '健身房全套'
);

-- 3. 训练记录（2周数据：2026-06-07 ~ 2026-06-20）
-- 采用推/拉/腿+肩的分化

-- 第1周：周一 胸+三头
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '杠铃卧推', 4, 10, 50, '热身组', '2026-06-08 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '杠铃卧推', 4, 8, 65, '正式组，最后一组力竭', '2026-06-08 10:10:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '哑铃飞鸟', 3, 12, 14, '', '2026-06-08 10:25:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '坐姿推胸', 3, 12, 40, '', '2026-06-08 10:35:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '绳索下压（三头）', 4, 12, 20, '', '2026-06-08 10:50:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '窄距卧推', 3, 10, 40, '', '2026-06-08 11:00:00');

-- 第1周：周三 背+二头
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '引体向上', 4, 6, 0, '自重，宽握', '2026-06-10 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '杠铃划船', 4, 10, 55, '弯腰90°，感受背阔肌', '2026-06-10 10:15:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '高位下拉', 3, 12, 45, '宽握', '2026-06-10 10:30:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '坐姿划船', 3, 12, 40, '窄握，挺胸', '2026-06-10 10:45:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '哑铃弯举', 4, 10, 12, '交替做', '2026-06-10 11:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '牧师凳弯举', 3, 12, 25, '', '2026-06-10 11:10:00');

-- 第1周：周五 腿
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '深蹲', 5, 8, 80, '注意膝盖不要内扣', '2026-06-12 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '腿举', 4, 10, 120, '', '2026-06-12 10:20:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '罗马尼亚硬拉', 3, 10, 60, '感受腘绳肌拉伸', '2026-06-12 10:35:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '腿屈伸', 3, 15, 40, '', '2026-06-12 10:50:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '站姿提踵', 4, 15, 30, '', '2026-06-12 11:00:00');

-- 第1周：周日 肩+腹
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '坐姿哑铃推举', 4, 8, 22, '肩部热身充分', '2026-06-14 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '哑铃侧平举', 4, 12, 8, '中束发力感不错', '2026-06-14 10:15:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '面拉', 4, 15, 15, '', '2026-06-14 10:30:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '悬垂举腿', 3, 12, 0, '', '2026-06-14 10:45:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '平板支撑', 3, 0, 0, '45秒', '2026-06-14 10:55:00');

-- 第2周：周一 胸+三头（重量略微增加，体现渐进超负荷）
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '杠铃卧推', 4, 8, 67.5, '比上周+2.5kg', '2026-06-15 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '哑铃飞鸟', 3, 12, 14, '', '2026-06-15 10:15:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '坐姿推胸', 3, 10, 45, '比上周+5kg', '2026-06-15 10:30:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '绳索下压（三头）', 4, 12, 22.5, '', '2026-06-15 10:45:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '窄距卧推', 3, 8, 42.5, '比上周+2.5kg', '2026-06-15 11:00:00');

-- 第2周：周三 背+二头
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '引体向上', 4, 7, 0, '比上周多1个', '2026-06-17 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '杠铃划船', 4, 8, 60, '比上周+5kg', '2026-06-17 10:15:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '高位下拉', 3, 10, 50, '', '2026-06-17 10:30:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '坐姿划船', 3, 10, 45, '', '2026-06-17 10:45:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '哑铃弯举', 4, 10, 14, '', '2026-06-17 11:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '锤式弯举', 3, 12, 12, '', '2026-06-17 11:10:00');

-- 第2周：周五 腿
INSERT INTO training_records (user_id, exercise, sets, reps, weight_kg, notes, recorded_at)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '深蹲', 5, 8, 85, '比上周+5kg，注意控制', '2026-06-19 10:00:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '腿举', 4, 8, 140, '比上周+20kg', '2026-06-19 10:20:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '罗马尼亚硬拉', 3, 10, 65, '', '2026-06-19 10:35:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '保加利亚分腿蹲', 3, 8, 20, '新加的动作', '2026-06-19 10:50:00'),
    ((SELECT id FROM users WHERE username = 'test_user'), '站姿提踵', 4, 15, 35, '', '2026-06-19 11:00:00');

-- 4. 训练反馈
INSERT INTO training_feedback (user_id, recorded_at, fatigue_level, pain_level, satisfaction, notes)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-08 11:15:00', 7, 1, 8, '今天卧推状态不错'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-10 11:20:00', 6, 0, 7, '背部发力感有进步'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-12 11:10:00', 8, 2, 9, '深蹲85kg突破，膝盖有点酸但能接受'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-14 11:00:00', 5, 0, 6, '肩部训练一般'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-15 11:10:00', 7, 1, 8, '加重量了，控制得还行'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-17 11:15:00', 6, 0, 8, '引体向上突破7个'),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-19 11:10:00', 8, 2, 9, '深蹲进步明显，腿举也加了');

-- 5. 身体测量数据（2周趋势）
INSERT INTO body_measurements (user_id, measured_at, weight_kg, body_fat_pct, chest_cm, waist_cm, arm_cm, thigh_cm)
VALUES
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-08 08:00:00', 72.0, 15.2, 100, 78, 37, 55),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-11 08:00:00', 72.3, 15.0, 100, 77.5, 37.5, 55.5),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-14 08:00:00', 72.5, 15.1, 101, 78, 37.5, 55.5),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-17 08:00:00', 72.8, 14.8, 101, 77.5, 38, 56),
    ((SELECT id FROM users WHERE username = 'test_user'), '2026-06-20 08:00:00', 73.0, 14.7, 102, 77, 38, 56);

-- 6. 训练计划
INSERT INTO training_plan (user_id, plan_name, start_date, end_date, plan_content)
VALUES (
    (SELECT id FROM users WHERE username = 'test_user'),
    '推/拉/腿分化 × 4天/周',
    '2026-06-08',
    '2026-07-06',
    '{
        "split": "Push/Pull/Legs + Shoulders",
        "weekly_schedule": {
            "Monday": "胸+三头",
            "Wednesday": "背+二头",
            "Friday": "腿部",
            "Sunday": "肩部+腹部"
        },
        "goals": "深蹲从80kg提升到100kg，卧推从65kg提升到75kg",
        "notes": "每两周增加一次重量（渐进超负荷），记录每次训练的感受"
    }'
);
