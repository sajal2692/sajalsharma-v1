module.exports = function(grunt) {

  // Project configuration.
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    
    watch: {

    },

    uglify: {
      my_target: {
        files: {
          'public/typed.min.js': ['public/typed.js'],
          'public/scripts.min.js': ['public/scripts.js']
        }
      }
    },
    
    cssmin: {

    },
    
    htmlmin: {

    }
    
  });

  // Load npms
  grunt.loadNpmTasks('grunt-contrib-uglify');
  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-contrib-cssmin');
  grunt.loadNpmTasks('grunt-contrib-htmlmin');
  grunt.loadNpmTasks('grunt-jekyll');

  // Default task(s).
  grunt.registerTask('default', ['uglify','htmlmin','cssmin']);

};