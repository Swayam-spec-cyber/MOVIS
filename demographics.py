def parse_gender_age(face):
    try:
        gender = face.sex
        gender_str = "Male" if gender == 1 else "Female"
        age = int(min(max(face.age, 1), 100))
        return gender_str, age
    except Exception as e:
        return "Unknown", "Unknown"
