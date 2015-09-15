/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos;
 *         Matriz de adjacencias.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Trabalho_Pratico_I
{

    /**
     * Classe que implementa um grafo direcionado, ou digrafo.
     */
    public class MatrizDigrafo : Digrafo
    {
        private int n;
        private int [,]adj;

        /**
         * Construtor de um digrafo de n vertices, sem nenhuma aresta.
         * As arestas devem ser adicionadas posteriormente.
          * Os vertices sao rotulados de 0 ate n-1.
         */
        public MatrizDigrafo(int n){
            this.n = n;
            this.adj = new int[n,n];
        }

        /**
         * Adiciona uma nova aresta direcionada do vertice u para v.
         */
        public override void adicionaAresta(int u, int v){
            this.adj[u,v] = 1;
        }

        /**
         * Remove uma nova aresta direcionada do vertice u para v, caso a mesma exista.
         */
        public override void removeAresta(int u, int v){
            this.adj[u,v] = 0;
        }

        /**
         * Verifica se existe uma aresta direcionada do vertice u para v.
         * @return true caso existir a aresta (u,v) ou false caso contrario.
         */
        public override Boolean existeAresta(int u, int v){
            if (this.adj[u,v] != 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * Obtem o numero de vertices do digrafo.
        * @return o numero de vertices.
         */
        public override int numVertices(){
            return this.n;
        }

        /**
         * Obtem o numero de arestas direcionadas do digrafo.
        * @return o numero de arestas do digrafo.
         */
        public override int numArestas(){
            int arestas = 0;
            for (int u = 0; u < numVertices(); u++)
            {
                for (int v = 0; v < numVertices(); v++)
                {
                    if (existeAresta(u, v))
                    {
                        arestas++;
                    }
                }
            }
            return arestas;
        }

        /**
        * Obtem os vizinhos de um dado vertice u.
        * @return conjunto de vizinhos do vertice u.
        */
        public override List<int> vizinhos(int u){
            List<int> resp = new List<int>();
            for (int v = 0; v < numVertices(); v++)
            {
                if (existeAresta(u, v))
                {
                    resp.Add(v);
                }
            }
            return resp;
        }

        /**
         * Verifica se existe algum loop no Digrafo.
        * @return true caso existir algum loop em qualquer vertice ou false caso contrario.
         */
        public override Boolean existeLoop(){
            //FAZER
            return false; //resposta temporaria
        }

        /**
         * Calcula o grau de saida do vertice u.
        * O grau de saida de um vertice u representa o numero de arestas direcionadas que possuem u como origem.
        * @return grau de saida do vertice u.
         */
        public override int grauSaida(int u){
            //FAZER
            return 0; //resposta temporaria
        }

        /**
         * Calcula o grau de entrada do vertice u.
        * O grau de entrada de um vertice u representa o numero de arestas direcionadas que possuem u como destino.
        * @return grau de entrada do vertice u.
         */
        public override int grauEntrada(int u){
            //FAZER
            return 0; //resposta temporaria
        }

		
    }
}
