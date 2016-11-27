#!/usr/bin/env python
if __name__ == '__main__':
    # Add src to $PYTHONPATH:
    from os.path import dirname, abspath
    from sys import path
    path.append(dirname(abspath(__file__)) + '/src/')

    # run classification pipeline
    from classification_pipeline_final import main
    main()
