﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Intex_2.Models
{
    public class EFCrashRepository : ICrashRepository
    {
        private CrashDbContext _context { get; set; }

        public EFCrashRepository(CrashDbContext temp)
        {
            _context = temp;
        }

        public IQueryable<Crash> Crashes => _context.Crashes;

        public IQueryable<Severity> Severities => _context.Severities;

        public void SaveCrash(Crash c)
        {
            _context.SaveChanges();
        }

        public void AddCrash(Crash c)
        {
            _context.Add(c);
            _context.SaveChanges();
        }

        public void DeleteCrash(Crash c)
        {
            _context.Remove(c);
            _context.SaveChanges();
        }
    }
}
