#ifndef __IMGIO_H__
#define __IMGIO_H__

#include <cstdlib>

#include <fstream>
using std::ifstream;
using std::ofstream;
using std::endl;

/*
    Função que lê o próximo caractere (char) do arquivo de entrada,
    ignorando espaço em branco e comentários de uma linha, #.

Parâmetros:
    in - arquivo de entrada, instância da classe ifstream.

Retorno:
     caractere alfanumérico lido do arquivo de entrada.
*/
char nextChar(ifstream &in)
{
    char ch;
    if(in){
        ch = in.get();

        do {
            //white space
            //ignora espaço em branco
            while( in && (ch==' ' || ch=='\n' || ch=='\t')  ) { ch = in.get(); }

            //comment
            //ignora comentários de uma linha, #
            if(ch=='#'){
                while(in && in.get()!='\n'){}
                ch = in.get();
            }//end if

        } while(in && !isalnum(ch));
    }
    return ch;
}

/*
    Função que lê o próximo inteiro(int) do arquivo ascii de entrada,
    ignorando espaço em branco e comentários de uma linha, #.

Parâmetros:
    in - arquivo de entrada no formato ascii, instância da classe ifstream.

Retorno:
     inteiro lido do arquivo ascii de entrada.
*/
int nextInt(ifstream &in)
{
    int val = 0;

    if(in){
        char temp[32];
        int i = 0;
        char ch;

        //ignora o caractere lido enquanto não for um digito
        while( in && !isdigit(ch=nextChar(in)) ){}

        //enquanto o caractere lido for um digito,
        //concatena-o ao buffer de caracteres
        while(in && isdigit(ch)){
            temp[i++] = ch;
            ch = in.get();
        }
        temp[i] = 0;

        //converte os caracteres de digitos lidos para um inteiro
        val = atoi(temp);
    }//end if

    //retorna o inteiro lido
    return val;
}


/*
    Formatos PPM, PGM.
    P2, P3
*/
Image loadPNMImage(const char *fileName)
{
    Image img;

    ifstream inFile(fileName);
    unsigned int max_value;
    char ch;

    if(inFile){

        //lê o cabeçalho do arquivo
        ch=nextChar(inFile);
        if(ch=='P'){
            ch=nextChar(inFile);
            if( ch=='3') {
                img.channel = 3;
            }else if( ch=='2') {
                img.channel = 1;
            }

            img.width = nextInt(inFile);
            img.height = nextInt(inFile);
            max_value = nextInt(inFile);
            if(!inFile) return img;

            /*
             após ler o cabeçalho do arquivo,
             cria-se uma imagem equivalente 'a
             imagem descrita pelo arquivo de entrada.
            */
            img.data = (int*)malloc(img.width*img.height*sizeof(int));

            int pixel;
            unsigned int i;
            unsigned int j;

            //lê os pixels da imagem
            for(i = 0; i<img.height && inFile; i++){
                for(j = 0; j<img.width && inFile; j++){
                    pixel = 0;
                    for(unsigned int k = 1; k<=img.channel && inFile; k++){
                        pixel = pixel | nextInt(inFile);
                        if( k<img.channel ) pixel = pixel<<8;
                    }
                    img.data[i*img.width+j] = pixel;
                }
            }

            //confere a validade do arquivo de entrada
            if(i<img.height || j<img.width){
                free(img.data);
            }
        }
    }


    //retorna a imagem lida
    //nulo se arquivo invalido
    return img;
}

/*
    Formatos PPM, PGM.
    P2, P3
*/
void storePNMImage(Image img, const char *fileName)
{
    ofstream outFile(fileName);

    if(outFile){

        char temp[25];
        char channelCode;
        int pixel;
        if(img.channel==3) channelCode = '3';
        else if(img.channel==1) channelCode = '2';

        //escreve o cabeçalho do arquivo
        outFile << 'P' << channelCode << endl;
        outFile << img.width << ' ' << img.height << endl;
        outFile << "255" << endl;

        //grava os pixels da imagem no arquivo de saída.
        for(unsigned int i = 0; i<img.height; i++){
            for(unsigned int j = 0; j<img.width; j++){
                pixel = img.data[i*img.width+j];
                for(unsigned int k = 1; k<=img.channel; k++){

                    unsigned char ch = (pixel>>(8*(img.channel-k)))&0xFF;

                    //itoa((int)ch, temp, 10);
                    sprintf(temp, "%d", int(ch));
                    outFile << temp << endl;
                }
            }
        }
        outFile.close();
    }
}

void filePNMPrint(FILE *out, Image img)
{
	if(out && img.data){
		char temp[25];
		char channelCode;
		int pixel;
		if(img.channel==3) channelCode = '3';
		else if(img.channel==1) channelCode = '2';

		//escreve o cabeçalho do arquivo
		fprintf(out, "P%d\n", channelCode);
		fprintf(out, "%d %d\n", img.width, img.height);
		fprintf(out, "255\n");

		//grava os pixels da imagem no arquivo de saída.
		for(unsigned int i = 0; i<img.height; i++){
			for(unsigned int j = 0; j<img.width; j++){
				pixel = img.data[i*img.width+j];
				for(unsigned int k = 1; k<=img.channel; k++){
					unsigned char ch = (pixel>>(8*(img.channel-k)))&0xFF;
						//itoa((int)ch, temp, 10);
						//sprintf(temp, "%d", int(ch));
						//outFile << temp << endl;
						fprintf(out, "%d\n", int(ch));
				}
			}
		}
	}
}


#endif
