#!/usr/bin/env python3
import os
import re
import sys

# Rutas absolutas de tus carpetas
DIRS = [
    '/home/lab/Desktop/TFG/images',
    '/home/lab/Desktop/TFG/poses'
]
OFFSET = 15

# Patrón: prefijo_ + dos dígitos + posible extensión
pattern = re.compile(r'^(?P<prefix>.+_)'
                     r'(?P<number>\d{2})'
                     r'(?P<ext>\..*)?$')

def rename_files():
    for dir_path in DIRS:
        if not os.path.isdir(dir_path):
            print(f"Error: '{dir_path}' no existe o no es un directorio.", file=sys.stderr)
            continue

        for fname in sorted(os.listdir(dir_path)):
            m = pattern.match(fname)
            if not m:
                # No coincide con *_NN o *_NN.ext
                continue

            prefix = m.group('prefix')
            number = m.group('number')
            ext = m.group('ext') or ''

            n = int(number)            # '00' → 0, '01' → 1, etc.
            new_n = n + OFFSET         # sumar 15
            new_number = f"{new_n:02d}"  # formatear a dos dígitos

            old_path = os.path.join(dir_path, fname)
            new_fname = f"{prefix}{new_number}{ext}"
            new_path = os.path.join(dir_path, new_fname)

            # Comprueba que no sobreescribas
            if os.path.exists(new_path):
                print(f"¡Salto! Ya existe: {new_path}", file=sys.stderr)
                continue

            # Renombra
            print(f"Renombrando:\n  {old_path}\n→ {new_path}")
            os.rename(old_path, new_path)

if __name__ == '__main__':
    rename_files()
