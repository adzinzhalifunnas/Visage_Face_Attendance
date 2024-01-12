import pathlib, livejson, face_attendance
from helpers.student_helper import StudentHelper

data_path = pathlib.Path('data')
db_path = "%s/faces" % data_path
student_data_path = pathlib.Path(f"{data_path}/students.json")

def main():
    student_data = livejson.File(student_data_path)
    student_helper = StudentHelper(student_data["students"])
    time_threshold = 2
    frame_threshold = 10
    face_attendance.start(db_path, student_helper, time_threshold=time_threshold, frame_threshold=frame_threshold)

if __name__ == "__main__":
    main()