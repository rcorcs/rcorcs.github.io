using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Trabalho_Pratico_I
{
    public class TestaDigrafo
    {

        public static String stringCorreto(int valor, int esperado)
        {
            return (esperado == valor) ? "OK" : "ERRADO";
        }

        public static String stringCorreto(Boolean valor, Boolean esperado)
        {
            return (esperado == valor) ? "OK" : "ERRADO";
        }

        public static void verifica(String texto, int valor, int esperado)
        {
            Console.WriteLine("{0}: {1} [{2}]\n", texto, valor, stringCorreto(valor, esperado));
        }

        public static void verifica(String texto, Boolean valor, Boolean esperado)
        {
            Console.WriteLine("{0}: {1} [{2}]\n", texto, valor, stringCorreto(valor, esperado));
        }


        public static void testaMatrizDigrafo()
        {
            Console.WriteLine("[Testando MatrizDigrafo]");
            Digrafo digrafo = new MatrizDigrafo(6);
            digrafo.adicionaAresta(0, 1);
            digrafo.adicionaAresta(0, 3);
            digrafo.adicionaAresta(1, 4);
            digrafo.adicionaAresta(2, 4);
            digrafo.adicionaAresta(2, 5);
            digrafo.adicionaAresta(3, 1);
            digrafo.adicionaAresta(4, 3);
            digrafo.adicionaAresta(5, 5);

            verifica("Numero vertices", digrafo.numVertices(), 6);
            verifica("Numero arestas", digrafo.numArestas(), 8);

            verifica("Existe aresta 0->1", digrafo.existeAresta(0, 1), true);
            verifica("Existe aresta 0->3", digrafo.existeAresta(0, 3), true);
            verifica("Existe aresta 1->4", digrafo.existeAresta(1, 4), true);
            verifica("Existe aresta 2->4", digrafo.existeAresta(2, 4), true);
            verifica("Existe aresta 2->5", digrafo.existeAresta(2, 5), true);
            verifica("Existe aresta 3->1", digrafo.existeAresta(3, 1), true);
            verifica("Existe aresta 4->3", digrafo.existeAresta(4, 3), true);
            verifica("Existe aresta 5->5", digrafo.existeAresta(5, 5), true);

            int arestasExistentes = 0;
            for (int u = 0; u < digrafo.numVertices(); u++)
            {
                for (int v = 0; v < digrafo.numVertices(); v++)
                {
                    if (digrafo.existeAresta(u, v))
                    {
                        arestasExistentes++;
                    }
                }
            }
            verifica("Arestas existentes", arestasExistentes, 8);

            verifica("Grau de saida vertice do 0", digrafo.grauSaida(0), 2);
            verifica("Grau de saida vertice do 1", digrafo.grauSaida(1), 1);
            verifica("Grau de saida vertice do 2", digrafo.grauSaida(2), 2);
            verifica("Grau de saida vertice do 3", digrafo.grauSaida(3), 1);
            verifica("Grau de saida vertice do 4", digrafo.grauSaida(4), 1);
            verifica("Grau de saida vertice do 5", digrafo.grauSaida(5), 1);

            verifica("Grau de entrada vertice do 0", digrafo.grauEntrada(0), 0);
            verifica("Grau de entrada vertice do 1", digrafo.grauEntrada(1), 2);
            verifica("Grau de entrada vertice do 2", digrafo.grauEntrada(2), 0);
            verifica("Grau de entrada vertice do 3", digrafo.grauEntrada(3), 2);
            verifica("Grau de entrada vertice do 4", digrafo.grauEntrada(4), 2);
            verifica("Grau de entrada vertice do 5", digrafo.grauEntrada(5), 2);

            verifica("Existe loop", digrafo.existeLoop(), true);

            Console.WriteLine("[Concluido MatrizDigrafo]");
            Console.ReadKey();
        }

        public static void testaListaDigrafo()
        {
            Console.WriteLine("[Testando ListaDigrafo]");
            Digrafo digrafo = new ListaDigrafo(6);
            digrafo.adicionaAresta(0, 1);
            digrafo.adicionaAresta(0, 3);
            digrafo.adicionaAresta(1, 4);
            digrafo.adicionaAresta(2, 4);
            digrafo.adicionaAresta(2, 5);
            digrafo.adicionaAresta(3, 1);
            digrafo.adicionaAresta(4, 3);
            digrafo.adicionaAresta(5, 5);

            verifica("Numero vertices", digrafo.numVertices(), 6);
            verifica("Numero arestas", digrafo.numArestas(), 8);

            verifica("Existe aresta 0->1", digrafo.existeAresta(0, 1), true);
            verifica("Existe aresta 0->3", digrafo.existeAresta(0, 3), true);
            verifica("Existe aresta 1->4", digrafo.existeAresta(1, 4), true);
            verifica("Existe aresta 2->4", digrafo.existeAresta(2, 4), true);
            verifica("Existe aresta 2->5", digrafo.existeAresta(2, 5), true);
            verifica("Existe aresta 3->1", digrafo.existeAresta(3, 1), true);
            verifica("Existe aresta 4->3", digrafo.existeAresta(4, 3), true);
            verifica("Existe aresta 5->5", digrafo.existeAresta(5, 5), true);

            int arestasExistentes = 0;
            for (int u = 0; u < digrafo.numVertices(); u++)
            {
                for (int v = 0; v < digrafo.numVertices(); v++)
                {
                    if (digrafo.existeAresta(u, v))
                    {
                        arestasExistentes++;
                    }
                }
            }
            verifica("Arestas existentes", arestasExistentes, 8);

            verifica("Grau de saida vertice do 0", digrafo.grauSaida(0), 2);
            verifica("Grau de saida vertice do 1", digrafo.grauSaida(1), 1);
            verifica("Grau de saida vertice do 2", digrafo.grauSaida(2), 2);
            verifica("Grau de saida vertice do 3", digrafo.grauSaida(3), 1);
            verifica("Grau de saida vertice do 4", digrafo.grauSaida(4), 1);
            verifica("Grau de saida vertice do 5", digrafo.grauSaida(5), 1);

            verifica("Grau de entrada vertice do 0", digrafo.grauEntrada(0), 0);
            verifica("Grau de entrada vertice do 1", digrafo.grauEntrada(1), 2);
            verifica("Grau de entrada vertice do 2", digrafo.grauEntrada(2), 0);
            verifica("Grau de entrada vertice do 3", digrafo.grauEntrada(3), 2);
            verifica("Grau de entrada vertice do 4", digrafo.grauEntrada(4), 2);
            verifica("Grau de entrada vertice do 5", digrafo.grauEntrada(5), 2);

            verifica("Existe loop", digrafo.existeLoop(), true);

            Console.WriteLine("[Concluido ListaDigrafo]");
            Console.ReadKey();
        }

        public static void Main(String[] args)
        {
            testaMatrizDigrafo();
            testaListaDigrafo();
        }
    }
}
