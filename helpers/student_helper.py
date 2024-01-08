class StudentHelper:
    def __init__(self, student):
        self.student = student

    def find_student_by_id(self, student_id):
        for student in self.student:
            if student["student_id"] == student_id:
                return student