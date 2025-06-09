"""
    Datasets generation tool

"""
import sys
import argparse
import parsy                    as P
from timeit                     import default_timer as timer
from datetime                   import timedelta
import os

# import commands.example         as cmd_example
# import commands.ula             as cmd_ula
# import commands.stl_conv             as cmd_stl
# import datasets.ura             as cmd_ura
# import datasets.piramide        as cmd_pyr
import commands.dataset_parallel  as cmd_ex

comandos = { 
        #  cmd_ula        : cmd_ula.command_description
        # ,cmd_stl        : cmd_ula.command_description
        # ,cmd_example    : cmd_example.command_description        
        # ,cmd_ura        : cmd_ura.command_description
        # ,cmd_pyr        : cmd_pyr.command_description
        cmd_ex         : cmd_ex.command_description
        }

description = 'Generador de Datasets'

def main(args):
    """
        main:: [String] -> IO None
    """
    base_parser = argparse.ArgumentParser(
           description   = description
          ,exit_on_error = False
    )

    front_parser = argparse.ArgumentParser(                                     # 1st parse with check cmdline againts
             parents    = [base_parser]                                         # base parser
            ,add_help   = False                                                 # show no mercy
            )
    front_parser.add_argument(                                                  # we are looking for the bloody command
            'cmd'                                                               # namespace key
            ,choices    =["'"+c.command_name for c in [*comandos.keys()]]           # one among these modules
            ,help       = 'Comando del dataset a ejecutar'                      # tell the dude what to do
            )   
    subparsers = base_parser.add_subparsers(
         title          = "Comandos disponibles"                            
        ,required       = True                                      
        ,dest           = 'cmd'                                                 # clave del NameSpace para el valor del módulo a correr
        ,description    = "Lista de comandos declarados en la herramienta (use --help para obtener ayuda de cada comando)"
        )
    
    print("ID del proceso del main.py: ", os.getpid())

    try:                                                                        # Try to...
        for module in comandos.keys():                                          # for each module  
            subparsers = module.add_subparser(subparsers)                       # register each module command parser
        
    except Exception as exc:                                                    # if something bad happened 
        print(f"{exc}\n")                                                       # show argparse error
        base_parser.print_help()                                                # run the module with appropieate options and test mode
    finally:                                                                    # Main program exit.
        print(f"Programa terminado")


if __name__ == "__main__":                                                      # if this module gets exe'd
    main(sys.argv[1:])                                                          # call the main entry point
