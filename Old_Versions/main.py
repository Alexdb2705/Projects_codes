"""
    Datasets generation tool

    - Oct 2024. First version, main code and example module.

"""
import sys
import argparse
import parsy                    as P
from timeit                     import default_timer as timer
from datetime                   import timedelta

# import commands.example         as cmd_example
# import commands.ula             as cmd_ula
# import commands.stl_conv             as cmd_stl
# import datasets.ura             as cmd_ura
# import datasets.piramide        as cmd_pyr
import commands.dataset_parallel_exe  as cmd_ex

comandos = { 
        #  cmd_ula        : cmd_ula.command_description
        # ,cmd_stl        : cmd_ula.command_description
        # ,cmd_example    : cmd_example.command_description        
        # ,cmd_ura        : cmd_ura.command_description
        # ,cmd_pyr        : cmd_pyr.command_description
        cmd_ex         : cmd_ex.command_description
        }

description = 'Generador de Datasets, @2024 Grupo de Electromagnetismo Computacional'

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
    
    try:                                                                        # Try to...
        for module in comandos.keys():                                          # for each module  
            subparsers = module.add_subparser(subparsers)                       # register each module command parser
        
        options  = base_parser.parse_args(args)                                 # now parse cmdline args to full modules specs

    except Exception as exc:                                                    # if something bad happened 
        print(f"{exc}\n")                                                       # show argparse error
        base_parser.print_help()                                                # help the user
    else:                                                                       # else command parsing went Ok
        comando(options.module, options, test = options.test)                   # run the module with appropieate options and test mode
    finally:                                                                    # Main program exit.
        print(f"Programa terminado")

def comando(module, args, test = False):
    """
        comando: Module -> Argparse.Namespace -> Bool -> IO ()
        Commands module runner
        Run the given module using the namespace parsed form command line switches
    """
    edition     = '#GenDat2024 V0.1 (CodeName: Johny Five)'
    start       = timer()
    try:
        print(f"{description} {edition}\n")
        if test:                                                                # if test suite solicited
            test_module(args, None)                                             # run test suite aborad the module
        else:                                                                   # else
            module.main(args)                                                   # run the module main subroutine
    except P.ParseError as PE:      print(f"Error found during parsing: {PE}")  # Error found parsing any file     
    except KeyboardInterrupt:       print(f"Program abort by Ctrl-C:")          # if user aborts bail out nicely 
    #except Exception as Exc:        print(f"Exception found: {Exc}")            # if an exception is caught bail out nicely 
    #except:                         print(f"Unknown error!")                    # Opps!
    finally:                                                                    # before finishing
        stop    = timer()                                                       # record time
        elapsed = timedelta(seconds = stop - start)                             # compute duration of the run         
        msg     = module.command_description                                    # what's the bussines inside
        print(f"\n{msg} terminó en {elapsed}\n\nHave a nice day!\n")              # tell da user
    return None

def test_module(args, _input):
    """
        test_module :: NameSpace -> None -> IO ()
        Run any test suite aboard the given module
    """
    import doctest                                                              # import doctest module
    (f, t) = doctest.testmod(                                                   # run test suite on the choosen module
             m          = args.module                                           # scan this module
            ,verbose    = False                                                 # supress chitchat 
            ,report     = True                                                  # show report
        )
    print(f"\n{f} tests fallidos de {t} tests definidos en '{args.module.__name__}'.\n")
    return None

if __name__ == "__main__":                                                      # if this module gets exe'd
    main(sys.argv[1:])                                                          # call the main entry point