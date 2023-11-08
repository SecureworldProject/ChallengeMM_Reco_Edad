# ChallengeMM_Reco_Edad
<h2>ChallengeMM_Reco_Edad</h2>
<p>
  El reconocimiento de edad es un proceso mediante el cual se utiliza la inteligencia artificial y algoritmos de visión por computadora para estimar la edad de una persona a partir de una imagen o video. Se analizan características faciales, como arrugas, líneas de expresión y tono de piel, para determinar un rango aproximado de edad. Esto tiene aplicaciones en áreas como la seguridad, la publicidad dirigida y la investigación demográfica.
</p>
<h3>descripcion de ficheros</h3>
<p>
  <ul>
<li>reconocimiento_edad.py:es un fichero que utiliza la cámara para capturar una foto y predecir el rango de edad de la persona en la imagen.
</li>
<li>"weights/opencv_face_detector.pbtxt": Este archivo contiene el modelo utilizado para la detección de caras.</li>
<li>"weights/opencv_face_detector_uint8.pb": En este archivo se encuentra el entrenamiento utilizado para la detección de caras.
</li>
<li>"weights/age_deploy.prototxt": Este archivo contiene el modelo utilizado para la detección de la edad de la persona en la imagen</li>
<li>"weights/age_net.caffemodel": En este archivo se encuentra el entrenamiento utilizado para calcular la edad de la persona detectada en la foto.</li>
  </ul>
</p>
<h3>modo de funcionamiento</h3>
<p>este challenge solo tiene un modo de funcionamiento, El parental ya que devuelve un 1 si es mayor de 16 años y un 0 si es menor</p>
