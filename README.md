Clonar repo https://github.com/mrl-amrl/DeepHAZMAT.git y modificar los directorios del codigo de donde se haya clonado el repo.

Modificar en el detector.py dentro del directorio DeepHAZMAT/deep_hazmat en la linea:
        
        self._layer_names = [self._layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

Y poner la siguiente linea:
        
        self._layer_names = [self._layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
