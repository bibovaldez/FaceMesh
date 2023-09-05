# facemesh detection
import cv2
import mediapipe as mp


class FaceMeshDetector():
    def __init__(self, maxFaces=2):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.facecon = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(imgRGB)
        if self.result.multi_face_landmarks:
            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms,
                                               self.mpFaceMesh.FACEMESH_CONTOURS, self.facecon, self.facecon)
        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    while cap.isOpened():
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        detector.findFaceMesh(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
