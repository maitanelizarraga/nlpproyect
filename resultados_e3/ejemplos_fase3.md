# Ejemplos de salidas (test)

## Texto 394

**Original:** ​​​​​​​​​​​​​​​​​​​​​​​​La Escuela Diplomática es una institución pública dependiente de la Subsecretaría ​del Ministerio de Asuntos Exteriores, Unión Europea y Cooperación de España. En la Escuela Diplomática se desarrolla el proceso selectivo para el ingreso en la Carrera Diplomática y la formación principal de los funcionarios de la Carrera Diplomática mediante la impartición de los cursos y materias procedentes para una formación continua.​

**Referencia:** La Escuela Diplomática es parte del Ministerio de Asuntos Exteriores de España. En este lugar se elige a las personas que quieren ser diplomáticos y se les forma para este trabajo. En la Escuela Diplomática se dan clases y se aprenden las cosas necesarias para ser un buen diplomático.

- **Baseline: copiar el original:** ​​​​​​​​​​​​​​​​​​​​​​​​La Escuela Diplomática es una institución pública dependiente de la Subsecretaría ​del Ministerio de Asuntos Exteriores, Unión Europea y Cooperación de España. En la Escuela Diplomática se desarrolla el proceso selectivo para el ingreso en la Carrera Diplomática y la formación principal de los funcionarios de la Carrera Diplomática mediante la impartición de los cursos y materias procedentes para una formación continua.​
- **Baseline: vecino más cercano (TF-IDF):** El Ministerio de Asuntos Exteriores, Unión Europea y Cooperación tiene cinco edificios importantes para llevar a cabo su trabajo: 
- Palacio de Viana
- Palacio de Santa Cruz
- Edificio Marqués de Salamanca (donde se encuentra la sede principal)
- Edificio de la AECID
- Escuela Diplomática
- **GRU+att+copia, Word2Vec propio fine-tuneado:** - escuela escuela es una institución encargada de la subsecretaría de ministerio de asuntos exteriores , unión europea y cooperación de españa . en la escuela diplomática se realiza en el escuela en la carrera diplomática y la formación principal de los funcionarios de la carrera . además , se han creado los cursos y materias procedentes para la formación continua.​ .

## Texto 430

**Original:** Las cookies son archivos que se pueden descargar en su equipo a través de las páginas web. Son herramientas que tienen un papel esencial para la prestación de numerosos servicios de la sociedad de la información. Entre otros, permiten a una página web almacenar y recuperar información sobre los hábitos de navegación de un usuario o de su equipo y, dependiendo de la información obtenida, se pueden utilizar para reconocer al usuario y mejorar el servicio ofrecido.

**Referencia:** - Las cookies son archivos que se descargan en tu computadora cuando visitas páginas web.
- Son importantes para que los sitios web puedan ofrecer diferentes servicios.
- Las cookies guardan información sobre cómo navegas en internet.
- Esta información ayuda a las páginas web a personalizar tu experiencia y mejorar el servicio que te ofrecen.

- **Baseline: copiar el original:** Las cookies son archivos que se pueden descargar en su equipo a través de las páginas web. Son herramientas que tienen un papel esencial para la prestación de numerosos servicios de la sociedad de la información. Entre otros, permiten a una página web almacenar y recuperar información sobre los hábitos de navegación de un usuario o de su equipo y, dependiendo de la información obtenida, se pueden utilizar para reconocer al usuario y mejorar el servicio ofrecido.
- **Baseline: vecino más cercano (TF-IDF):** - Las cookies son archivos que se guardan en tu dispositivo cuando visitas páginas web.
- Son útiles para que las páginas web puedan ofrecer diferentes servicios.
- Las cookies guardan información sobre cómo navegas en internet.
- Esta información puede ayudar a que la página te reconozca y mejore el servicio que te ofrece.
- **GRU+att+copia, Word2Vec propio fine-tuneado:** - las cookies son archivos que se descargan en tu computadora cuando visitas páginas web . son útiles para brindar diferentes servicios en internet . por ejemplo , permiten que una página web guarde información sobre cómo navegas en ella . - esta información se usa para reconocerte y mejorar el servicio que te ofrecen .

## Texto 41

**Original:** Según quien sea la entidad que gestione el equipo o dominio desde donde se envían las cookies y trate los datos que se obtengan, podemos distinguir: Cookies propias: Son aquellas que se envían al equipo terminal del usuario desde www.exteriores.gob.es. Cookies de terceros: Son aquellas que se envían al equipo terminal del usuario desde un equipo o dominio que no es gestionado por el editor, sino por otra entidad que trata los datos obtenidos través de las cookies, como www.google.com.

**Referencia:** Las cookies pueden ser de dos tipos, dependiendo de quién las envíe y gestione los datos que se recopilan:

1. Cookies propias: Son enviadas desde www.exteriores.gob.es al dispositivo del usuario. 
   
2. Cookies de terceros: Son enviadas desde un sitio web diferente al de www.exteriores.gob.es, como www.google.com. Esta información es manejada por otra entidad distinta al sitio web que estás visitando.

- **Baseline: copiar el original:** Según quien sea la entidad que gestione el equipo o dominio desde donde se envían las cookies y trate los datos que se obtengan, podemos distinguir: Cookies propias: Son aquellas que se envían al equipo terminal del usuario desde www.exteriores.gob.es. Cookies de terceros: Son aquellas que se envían al equipo terminal del usuario desde un equipo o dominio que no es gestionado por el editor, sino por otra entidad que trata los datos obtenidos través de las cookies, como www.google.com.
- **Baseline: vecino más cercano (TF-IDF):** **Cookies analíticas en www.exteriores.gob.es:**

- **¿Qué son las cookies analíticas?**
Las cookies analíticas son aquellas que nos ayudan a saber cuántas personas visitan nuestra página web y nos permiten estudiar cómo la utilizan. Esto nos ayuda a mejorar nuestro sitio web.

- **¿Qué tipos de cookies analíticas usamos en www.exteriores.gob.es?**
En nuestra web utilizamos cookies analíticas propias, de sesión y persistentes. En concreto, usamos cookies de Google Analytics, que tienen diferentes tiempos de duración dependiendo de la información que nos proporcionan.

- **¿Para qué sirven estas cookies?**
Estas cookies nos ayudan a entender cómo las personas navegan en nuestra página y qué podemos hacer para hacerla mejor. También nos permiten hacer mediciones y análisis estadísticos para mejorar la experiencia de usuario.

- **¿Quiénes más utilizan este tipo de cookies en nuestra web?**
Además de nosotros, otras empresas como Google, Youtube o Twitter también utilizan cookies analíticas de sesión y persistentes en nuestra página.
- **GRU+att+copia, Word2Vec propio fine-tuneado:** aquí te dejo la versión en lenguaje claro : 1 . * * ¿ qué qué son las cookies ? * * - las cookies son pequeños archivos que se guardan en tu computadora cuando visitas páginas web . 2 . las cookies propias son archivos que guardan información en tu dispositivo cuando visitas un sitio web . 3 . ¿ qué cookies propias cookies :* * - ayudan a que las páginas web funcione correctamente . - * * cookies terminal :* * : son las equipo de terceros . - las aquellas son archivos para que el sitio web funcione bien . - es importante saber que la página funcione correctamente , pero no dudes en la editor .

## Texto 265

**Original:** El ministro de Transportes y Movilidad Sostenible, Óscar Puente, ha presidido hoy la inauguración del muelle de Raos 9, en el puerto de Santander, donde ha destacado la importancia de esta nueva estructura para impulsar la actividad y competitividad del Puerto, afianzándolo como polo logístico para las exportaciones e importaciones de vehículos.

**Referencia:** Hoy, el ministro de Transporte, Óscar Puente, estuvo en la apertura del muelle de Raos 9 en el puerto de Santander. Habló sobre lo importante que es esta nueva parte del puerto para ayudar a que el lugar sea más activo y competitivo. Esto hará que el puerto sea un lugar clave para enviar y recibir coches.

- **Baseline: copiar el original:** El ministro de Transportes y Movilidad Sostenible, Óscar Puente, ha presidido hoy la inauguración del muelle de Raos 9, en el puerto de Santander, donde ha destacado la importancia de esta nueva estructura para impulsar la actividad y competitividad del Puerto, afianzándolo como polo logístico para las exportaciones e importaciones de vehículos.
- **Baseline: vecino más cercano (TF-IDF):** Hoy, el ministro de Transporte y Movilidad Sostenible, Óscar Puente, ha abierto oficialmente el muelle de Raos 9 en el puerto de Santander. Destacó que esta nueva parte es muy importante para ayudar al Puerto a ser más activo y competitivo, convirtiéndose en un lugar clave para enviar y traer coches y otros vehículos al país.
- **GRU+att+copia, Word2Vec propio fine-tuneado:** hoy , el ministro de transportes y movilidad sostenible , óscar puente , ha liderado la inauguración del muelle de raos 9 en el puerto de santander . destacó que la muelle es muy importante para ayudar a que el puerto sea más activo y competitivo .

## Texto 523

**Original:** El portal web del Ministerio para la Transición Ecológica y el Reto Demográfico utiliza Google Analytics, esta es una herramienta de analítica que ayuda a los sitios web y a los propietarios de aplicaciones a entender el modo en que sus visitantes interactúan con sus propiedades. Se utilizan un conjunto de cookies para recopilar información e informar de las estadísticas de uso de los sitios web sin identificar personalmente a los visitantes de Google.

**Referencia:** El sitio web del Ministerio de Transición Ecológica y Reto Demográfico usa Google Analytics, que es una herramienta para entender cómo la gente usa el sitio. Utiliza cookies para recopilar datos y mostrar estadísticas sin identificar a los visitantes de forma personal.

- **Baseline: copiar el original:** El portal web del Ministerio para la Transición Ecológica y el Reto Demográfico utiliza Google Analytics, esta es una herramienta de analítica que ayuda a los sitios web y a los propietarios de aplicaciones a entender el modo en que sus visitantes interactúan con sus propiedades. Se utilizan un conjunto de cookies para recopilar información e informar de las estadísticas de uso de los sitios web sin identificar personalmente a los visitantes de Google.
- **Baseline: vecino más cercano (TF-IDF):** - El sitio web del Ministerio para la Transición Ecológica y el Reto Demográfico usa una herramienta llamada Google Analytics. Esta herramienta ayuda a los dueños de sitios web a entender cómo interactúan los visitantes con sus páginas.
- Google Analytics utiliza cookies para recopilar información y mostrar estadísticas sobre el uso de los sitios web, pero esta información no identifica a los visitantes de forma personal.
- **GRU+att+copia, Word2Vec propio fine-tuneado:** el sitio web del ministerio para la transición ecológica y el reto demográfico usa una herramienta llamada google analytics . esta herramienta ayuda a entender cómo interactúan los visitantes con el sitio . utiliza cookies para recopilar información y mostrar estadísticas de uso del sitio web , pero no identifica personalmente a los visitantes de google .

## Texto 497

**Original:** Las cookies son archivos que se pueden descargar en su equipo a través de las páginas web. Son herramientas que tienen un papel esencial para la prestación de numerosos servicios de la sociedad de la información. Entre otros, permiten a una página web almacenar y recuperar información sobre los hábitos de navegación de un usuario o de su equipo y, dependiendo de la información obtenida, se pueden utilizar para reconocer al usuario y mejorar el servicio ofrecido.

**Referencia:** - Las cookies son archivos que las páginas web pueden poner en tu computadora.
- Ayudan a ofrecer diferentes servicios en internet.
- Las cookies guardan información sobre cómo usas internet.
- Esta información permite que la página te reconozca y mejore su servicio.

- **Baseline: copiar el original:** Las cookies son archivos que se pueden descargar en su equipo a través de las páginas web. Son herramientas que tienen un papel esencial para la prestación de numerosos servicios de la sociedad de la información. Entre otros, permiten a una página web almacenar y recuperar información sobre los hábitos de navegación de un usuario o de su equipo y, dependiendo de la información obtenida, se pueden utilizar para reconocer al usuario y mejorar el servicio ofrecido.
- **Baseline: vecino más cercano (TF-IDF):** - Las cookies son archivos que se guardan en tu dispositivo cuando visitas páginas web.
- Son útiles para que las páginas web puedan ofrecer diferentes servicios.
- Las cookies guardan información sobre cómo navegas en internet.
- Esta información puede ayudar a que la página te reconozca y mejore el servicio que te ofrece.
- **GRU+att+copia, Word2Vec propio fine-tuneado:** - las cookies son archivos que se descargan en tu computadora cuando visitas páginas web . son útiles para brindar diferentes servicios en internet . por ejemplo , permiten que una página web guarde información sobre cómo navegas en ella . - esta información se usa para reconocerte y mejorar el servicio que te ofrecen .
